import struct

from ...java.jni.reference import *

from ...java.jni.constants import *
from ...java.jvm.constants import JAVA_NULL

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from ...java.jni.main import JNIEnv
    from unicorn import Uc

# JNIEnv tools

# JNI type names that hold a reference (resolved through the JNIEnv reference tables).
_REF_TYPE_NAMES = frozenset(('jobject', 'jstring', 'jclass', 'jthrowable', 'jarray', 'jweak'))
_INT_TYPE_NAMES = frozenset(('jboolean', 'jbyte', 'jchar', 'jshort', 'jint'))


def _arg_kind(t: str) -> str:
    """
    Classifies an args_list type name for the register / va_list readers:
    'l' jlong, 'd' jdouble, 'f' jfloat, 'o' reference, 'i' int-like.

    The readers used to look at t[1] only, which cannot tell jshort from jstring (both 's',
    so a short was looked up as a reference) and misses jclass / jarray / jbyteArray / ...
    (read as plain ints). The known JNI names are matched in full; any other name keeps the old
    first-letter behaviour so existing custom args_list values do not change meaning.
    """
    if t == 'jlong': return 'l'
    if t == 'jdouble': return 'd'
    if t == 'jfloat': return 'f'
    if t in _REF_TYPE_NAMES or t.endswith('Array'): return 'o'
    if t in _INT_TYPE_NAMES: return 'i'

    c = t[1]
    if c == 's': return 'o'
    return c


def count_arg_words32(args_type_list: List, first_reg: int = 3) -> int:
    """
    How many 32 bit words the (variadic) arguments occupy on ARM32, counted from the first one:
    a jlong/jdouble takes two words and starts at an even register/stack position, so the word
    before it may be left unused - exactly what read_args32 expects to find in its `args`.
    `first_reg` is the register number of the first variadic argument (3 after env, obj, methodID).
    """
    idx = 0
    for t in args_type_list or ():
        if _arg_kind(t) in ('l', 'd'):
            idx += (first_reg + idx) & 1
            idx += 2
        else:
            idx += 1
    return idx

def read_args32(jni: 'JNIEnv', mu: 'Uc', args: List, args_type_list: List) -> List:
    if not args_type_list: return []
    
    result = []
    idx = 0
    n = len(args)
    get_ref = jni.get_reference

    for t in args_type_list:
        if idx >= n: break
        
        c = _arg_kind(t)
        if c == 'l' or c == 'd': # jlong, jdouble
            idx += (3 + idx) & 1
            if idx + 1 >= n: break
            
            v = (args[idx+1] << 32) | (args[idx] & 0xFFFFFFFF)
            if c == 'l' and v >> 63: v -= 1 << 64
            
            result.append(v)
            idx += 2
        elif c == 'o': # jstring, jobject, ...
            ref = get_ref(args[idx])
            result.append(JAVA_NULL if ref is None else ref.value)
            idx += 1
        else: # jint, jchar, jbyte, jboolean
            result.append(args[idx])
            idx += 1
            
    return result

def read_args64(jni: 'JNIEnv', mu: 'Uc', args: List, args_type_list: List) -> List:
    if not args_type_list: return []

    result = []
    get_ref = jni.get_reference
    for i in range(len(args_type_list) if len(args_type_list) < len(args) else len(args)):
        t = args_type_list[i]
        v = args[i]
        
        # references: jstring, jobject, ...
        if _arg_kind(t) == 'o':
            ref = get_ref(v)
            result.append(JAVA_NULL if ref is None else ref.value)
        else:
            result.append(v)
            
    return result

def read_args_v32(jni: 'JNIEnv', mu: 'Uc', args_ptr, args_type_list) -> List:
    if not args_type_list: return []
    
    res = []
    p = args_ptr
    get_ref = jni.get_reference
    read = mu.mem_read

    for t in args_type_list:
        c = _arg_kind(t)
        if c == 'l' or c == 'd': # jlong, jdouble
            p += p & 4
            v = struct.unpack('<q' if c == 'l' else '<Q', read(p, 8))[0]
            res.append(v)
            p += 8
        elif c == 'o': # jstring, jobject, ...
            v = struct.unpack('<I', read(p, 4))[0]
            ref = get_ref(v)
            res.append(JAVA_NULL if ref is None else ref.value)
            p += 4
        else: # jint, jchar, jbyte, jboolean
            res.append(struct.unpack('<I', read(p, 4))[0])
            p += 4
            
    return res

def read_args_v64(jni: 'JNIEnv', mu: 'Uc', args_ptr: int, args_type_list: List) -> List:
    if not args_type_list: return []
    
    # __stack(Q), __gr_top(Q), __vr_top(Q), __gr_offs(i), __vr_offs(i)
    stk, gr_t, vr_t, gr_o, vr_o = struct.unpack("<QQQii", mu.mem_read(args_ptr, 32))
    
    res = []
    read = mu.mem_read
    get_ref = jni.get_reference

    for t in args_type_list:
        c = _arg_kind(t)
        # jdouble (d) jfloat (f) using VR
        if c == 'd' or c == 'f':
            if vr_o >= 0:
                addr = stk
                stk += 8
            else:
                addr = vr_t + vr_o
                vr_o += 16 # VR args in AAPCS64 using 16 bytes (Q-reg)
            
            v = struct.unpack("<Q", read(addr, 8))[0]
            res.append(v)
        else:
            # (int, long, object, etc)
            if gr_o >= 0:
                addr = stk
                stk += 8
            else:
                addr = gr_t + gr_o
                gr_o += 8
            
            v = struct.unpack("<Q", read(addr, 8))[0]
            
            if c == 'o': # jstring, jobject, ...
                if v == 0:
                    res.append(JAVA_NULL)
                else:
                    ref = get_ref(v)
                    res.append(JAVA_NULL if ref is None else ref.value)
            else:
                res.append(v)
                
    return res

# jvalue[] (the "A" variants: Call<Type>MethodA, CallStatic<Type>MethodA, NewObjectA, ...)
#
# jvalue is a union, so every element of the array occupies 8 bytes on both ARM32 and ARM64
# and a caller only writes the bytes of the member it uses: for a jint the upper 4 bytes of
# the slot are garbage, for a jboolean the upper 7 bytes. That is why each argument is
# decoded with its exact width (little endian) instead of reading the whole slot. Unlike the
# va_list variants nothing is passed in registers here, so there is no GPR/VR split to follow.
#
# Values are given to the mocks with Java semantics: signed byte/short/int/long, unsigned
# char, jboolean as 0/1, jfloat/jdouble as Python float, references resolved to the objects
# behind them (JAVA_NULL for null).
_JVALUE_SIZE = 8

_JVALUE_FMT = {
    'jboolean': '<B',
    'jbyte': '<b',
    'jchar': '<H',
    'jshort': '<h',
    'jint': '<i',
    'jlong': '<q',
    'jfloat': '<f',
    'jdouble': '<d',
}

_JVALUE_REF_TYPES = frozenset(('jobject', 'jstring', 'jclass', 'jthrowable', 'jarray', 'jweak'))


def _read_args_a(jni: 'JNIEnv', mu: 'Uc', args_ptr: int, args_type_list: List, ref_fmt: str) -> List:
    # A method without arguments may be called with a NULL (or garbage) jvalue pointer.
    if not args_type_list: return []

    raw = mu.mem_read(args_ptr, _JVALUE_SIZE * len(args_type_list))
    res = []
    get_ref = jni.get_reference

    for i, t in enumerate(args_type_list):
        off = i * _JVALUE_SIZE
        fmt = _JVALUE_FMT.get(t)
        if fmt is not None:
            v = struct.unpack_from(fmt, raw, off)[0]
            res.append((1 if v else 0) if t == 'jboolean' else v)
        elif t in _JVALUE_REF_TYPES or t.endswith('Array'): # jstring, jobject, jbyteArray, ...
            v = struct.unpack_from(ref_fmt, raw, off)[0]
            ref = get_ref(v)
            res.append(JAVA_NULL if ref is None else ref.value)
        else:
            raise ValueError(
                f"Unsupported argument type {t!r} in args_list for a jvalue[] call "
                f"(use one of: {', '.join(sorted(_JVALUE_FMT))}, {', '.join(sorted(_JVALUE_REF_TYPES))})"
            )

    return res

def read_args_a32(jni: 'JNIEnv', mu: 'Uc', args_ptr: int, args_type_list: List) -> List:
    # jobject is a 4 byte pointer, the upper half of its slot is not meaningful
    return _read_args_a(jni, mu, args_ptr, args_type_list, '<I')

def read_args_a64(jni: 'JNIEnv', mu: 'Uc', args_ptr: int, args_type_list: List) -> List:
    return _read_args_a(jni, mu, args_ptr, args_type_list, '<Q')
