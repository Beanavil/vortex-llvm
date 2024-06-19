# This test generates all variants of wmma intrinsics and verifies that LLVM
# generates correct instructions for them.

# Check all variants of instructions supported by PTX60 on SM70
# RUN: %python %s --ptx=60 --gpu-arch=70 > %t-ptx60-sm_70.ll
# RUN: FileCheck %t-ptx60-sm_70.ll < %t-ptx60-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,M16N16
# RUN: FileCheck %t-ptx60-sm_70.ll < %t-ptx60-sm_70.ll \
# RUN:           --check-prefixes=INTRINSICS,NOEXTGEOM,NOINT,NOSUBINT,NOMMA,NODOUBLE,NOALTFLOAT,NOLDMATRIX
# RUN: llc < %t-ptx60-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 \
# RUN:           | FileCheck %t-ptx60-sm_70.ll
# RUN: %if ptxas %{                                                       \
# RUN:   llc < %t-ptx60-sm_70.ll -march=nvptx64 -mcpu=sm_70 -mattr=+ptx60 \
# RUN:           | %ptxas-verify -arch=sm_70                              \
# RUN: %}

from __future__ import print_function

import argparse
from itertools import product
from string import Template

class MMAType:
  def __init__(self, s_type):
    self.s_type = s_type
    self.llvm_type = {
        "s32"  : "i32",
        "u32"  : "i32",
    }[s_type];

    # self.ptx_reg_pattern = {
    #     "f16" : "%hh[0-9]+",
    #     "f32" : "%f[0-9]+",
    #     "f64" : "%fd[0-9]+",
    # }.get(s_type, "%r[0-9]+")

  def __repr__(self):
    return "%s/%s" % (self.s_type, self.llvm_type)

class MMAFrag:
  def __init__(self, geom, frag, elt_type):
    self.geom = geom
    self.frag = frag
    self.mma_type = MMAType(elt_type);
    self.nregs = {
        "m2n2k2:a:u32" : 2,
        "m2n2k2:a:s32" : 2,
        "m2n2k2:b:u32" : 2,
        "m2n2k2:b:s32" : 2,
        "m2n2k2:c:u32" : 1,
        "m2n2k2:c:s32" : 1,
    }.get("%s:%s:%s" % (geom, frag, elt_type))
    assert(self.nregs);

  def __repr__(self):
    return "%s:%s:%s%s" % (self.geom, self.frag, self.mma_type,
                           "" if self.nregs == 1 else ("*%d" % self.nregs))

class MMAOp:
  def __init__(self, a, b, c):
    self.a = a
    self.b = b
    self.c = c

  def __repr__(self):
    return ("{A:%s, B:%s, C:%s}" % (self.a, self.b, self.c))

def make_mmul_ops(geoms, types_a, types_b, types_c):
  ops = []
  for geom, type_a, type_c in product(geoms, types_a, types_c):
    for type_b in product(types_b if types_b else [type_a]):
      ops.append(MMAOp(MMAFrag(geom, "a", type_a),
                       MMAFrag(geom, "b", type_b),
                       MMAFrag(geom, "c", type_c)))
  return ops

def make_mldst_ops(geoms, frags, types):
  return [MMAFrag(geom, frag, s_type) for (geom, frag, s_type)
          in product(geoms, frags, types)]

def get_mma_ops():
  return (make_mmul_ops(["m2n2k2", "m4n4k4"],
                       ["s32", "u32"], [], ["s32", "u32"]))

def get_ldst_ops(kind):
  ldst_ops = (make_mldst_ops(["m2n2k2", "m4n4k4"],
                            ["a", "b", "c"], ["s32", "u32"]))
  return [ x for x in ldst_ops if (x.frag == "c") == (kind == "store")]

def get_ldmatrix_ops():
  return make_ldmatrix_ops(["m8n8"], ["x1", "x2", "x4"], ["b16"])

def make_m_slice_ty(frag):
  return [frag.mma_type.llvm_type] * frag.nregs

def make_m_ld_ret_ty(frag):
  results = make_m_slice_ty(frag)
  if len(results) == 1:
    return "%s" % results[0]
  return "{%s}" % ", ".join(results)

def check_pattern(frag):
   return "{{%s}}" % ", *".join([frag.mma_type.ptx_reg_pattern] * frag.nregs)

def gen_m_load_tests():
  load_template = """
declare ${ret_ty} @${intrinsic}(i8 ${as}* %src ${extra_args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define ${ret_ty} @test_${function}(i8 ${as}* %src ${extra_args}) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}]${stride_pattern}
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src ${extra_args});
  ret ${ret_ty} %v0;
}

; CHECK-LABEL: .func{{.*}}test_${function}_o(
define ${ret_ty} @test_${function}_o(i8 ${as}* %src ${extra_args}) {
; CHECK: ${instruction}
; CHECK: {${check_result}}
; CHECK: [%rd{{[0-9]+}}+128]${stride_pattern}
  %src1 = getelementptr i8, i8 ${as}* %src, i32 128;
  %v0 = call ${ret_ty} @${intrinsic}(i8 ${as}* %src1 ${extra_args});
  ret ${ret_ty} %v0;
}
"""
  intrinsic_template = "vx_mload_${geom}_${abc}_${itype}_${stride}"
  instruction_template = "int_riscv_vx_mload_${geom}_${abc}_${itype}_${stride}"

  generated_items = []

  for frag, stride in product(
      get_ldst_ops("load"),
      ["", ".stride"],
      ):

    params = {
        "abc" : frag.frag,
        "stride" : stride,
        "itype" : frag.mma_type.s_type,
        "geom"   : frag.geom,
    }

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".","_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_m_ld_ret_ty(frag)
    test_params["check_result"] = check_pattern(frag)

    if stride:
      test_params["extra_args"] = ", i32 %stride";
      test_params["stride_pattern"] = ", %r{{[0-9]+}}"
    else:
      test_params["extra_args"] = ""
      test_params["stride_pattern"] = ""

    print(Template(load_template).substitute(test_params))

    generated_items.append((test_params["intrinsic"],
                            test_params["instruction"]))

  return generated_items

def make_m_slice_args(frag):
  return ", ".join(["%s %%%s%d" % (t, frag.frag, i) for i,t
                  in enumerate(make_m_slice_ty(frag))])

def gen_m_store_tests():
  store_template = """
declare void @${intrinsic}(i8 ${as}* %src, ${args}${extra_args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define void @test_${function}(i8 ${as}* %src, ${args}${extra_args}) {
; CHECK: ${instruction} {{.*}}[%rd{{[0-9+]}}
; CHECK: {${check_args}}
; CHECK: ${stride_pattern}
  call void @${intrinsic}(i8 ${as}* %src, ${args} ${extra_args});
  ret void
}

; CHECK-LABEL: .func{{.*}}test_${function}_o(
define void @test_${function}_o(i8 ${as}* %src, ${args}${extra_args}) {
; CHECK: ${instruction} {{.*}}[%rd{{[0-9+]}}+128]
; CHECK: ${check_args}
; CHECK: ${stride_pattern}
  %src1 = getelementptr i8, i8 ${as}* %src, i32 128;
  call void @${intrinsic}(i8 ${as}* %src1, ${args}${extra_args});
  ret void
}
"""
  intrinsic_template = "vx_mstore_${geom}_${abc}_${itype}_${stride}"
  instruction_template = "int_riscv_vx_mstore_${geom}_${abc}_${itype}_${stride}"

  generated_items = []

  for frag, layout, space, stride in product(
      get_ldst_ops("store"),
      ["", ".stride"]):

    params = {
        "abc" : frag.frag,
        "stride" : stride,
        "itype" : frag.mma_type.s_type,
        "geom"   : frag.geom,
    }

    test_params = params
    test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
    test_params["function"] = test_params["intrinsic"].replace(".","_")
    test_params["instruction"] = Template(instruction_template).substitute(params)
    test_params["ret_ty"] = make_m_ld_ret_ty(frag)
    test_params["check_args"] = check_pattern(frag)
    if stride:
      test_params["extra_args"] = ", i32 %stride";
      test_params["stride_pattern"] = ", %r{{[0-9]+}};"
    else:
      test_params["extra_args"] = ""
      test_params["stride_pattern"] = ";"
    test_params["args"] = make_m_slice_args(frag);

    print(Template(store_template).substitute(test_params))
    generated_items.append((test_params["intrinsic"],
                            test_params["instruction"]))

  return generated_items

def mmul_signature(op):
  return op.a.mma_type.s_type

def mmul_s_signature(op):
  # Encode all three types as C.A.B
  return ".".join(x.mma_type.s_type for x in (op.c, op.a, op.b))

def common_mmul_test_gen(params, op, intrinsic_template, instruction_template):
  mma_template = """
declare ${ret_ty} @${intrinsic}(
        ${args});

; CHECK-LABEL: .func {{.*}}test_${function}(
define ${ret_ty} @test_${function}(
        ${args}) {
; CHECK: ${instruction}
; CHECK-NEXT: ${check_d}
; CHECK-NEXT: ${check_a}
; CHECK-NEXT: ${check_b}
; CHECK-NEXT: ${check_c}
  %r = call ${ret_ty} @${intrinsic}(
        ${args});
  ret ${ret_ty} %r;
}
"""

  test_params = params
  test_params["intrinsic"] = Template(intrinsic_template).substitute(params)
  test_params["function"] = test_params["intrinsic"].replace(".", "_")
  test_params["instruction"] = Template(instruction_template).substitute(params)
  test_params["ret_ty"] = make_m_ld_ret_ty(op.d)
  test_params["check_a"] = check_pattern(op.a)
  test_params["check_b"] = check_pattern(op.b)
  test_params["check_c"] = check_pattern(op.c)
  args = ",\n        ".join(make_m_slice_args(frag)
                            for frag in (op.a, op.b))
  test_params["args"] = args
  print(Template(mma_template).substitute(test_params))
  return (test_params["intrinsic"], test_params["instruction"])

def gen_mmul_tests():
  mma_intrinsic_template = "vx_mmul_${geom}_${intrinsic_signature}"
  mma_instruction_template = "int_riscv_vx_mmul_${geom}_${intrinsic_signature}"

  generated_items=[]

  for op in product(get_mma_ops()):
    params = {
        "intrinsic_signature" : mmul_signature(op),
        "ptx_signature" : mmul_s_signature(op),
        "geom"  : op.a.geom,
    }

    intrinsic_template = mma_intrinsic_template
    instruction_template = mma_instruction_template

    generated_items.append(common_mmul_test_gen(params, op,
      intrinsic_template, instruction_template))

  return generated_items

# Append complete list of intrinsics and instructions we've generated tests for.
# Generate set of checks to verify that that we did generate sensible set of
# tests for the given combination of PTX and SM variants.
#
def gen_check_unsupported_ops(items):
  print("; Complete list of intrinsics supported by PTX%d on sm_%d"
        % (ptx_version, gpu_arch))
  print("; INTRINSICS: {{^; INTRINSICS_LIST_BEGIN}}")
  print("""

; NOEXTGEOM-NOT: {{m8n32|m32n8}}
; NOINT-NOT: .{{s32|s8}}
; NOSUBINT-NOT: {{s4|u4|b1}}
; NOMMA-NOT: .m8n8k4.
; NOALTFLOAT-NOT: .{{bf16|tf32}}
; NODOUBLE-NOT: .f64
; NOLDMATRIX-NOT: ldmatrix.sync.aligned

; M16N16-DAG: m16n16k16.load.{{[ab].*}}.f16.p
; M16N16-DAG: m16n16k16.{{load|store}}.{{[cd].*\.(f16|f32)}}.p
; M16N16-DAG: m16n16k16.mma.{{.*}}.f16.f32
; M16N16-DAG: m16n16k16.mma.{{.*}}.f32.f16
; M16N16-DAG: m16n16k16.mma.{{.*}}.f16.f16
; M16N16-DAG: m16n16k16.mma.{{.*}}.f32.f32

; PTX60 adds support for m32n8k16/m8n32k16 geometries.
; EXTGEOM-DAG: m32n8k16.load.{{[ab].*}}.f16.p
; EXTGEOM-DAG: m32n8k16.{{load|store}}.{{[cd].*\.(f16|f32)}}.p
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f16.f32
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f32.f16
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f16.f16
; EXTGEOM-DAG: m32n8k16.mma.{{.*}}.f32.f32

; EXTGEOM-DAG: m8n32k16.load.{{[ab].*}}.f16.p
; EXTGEOM-DAG: m8n32k16.{{load|store}}.{{[cd].*\.(f16|f32)}}.p
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f16.f32
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f32.f16
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f16.f16
; EXTGEOM-DAG: m8n32k16.mma.{{.*}}.f32.f32

; INT-DAG: m16n16k16.load.{{[ab].*}}.s8.p
; INT-DAG: m8n32k16.load.{{[ab].*}}.s8.p
; INT-DAG: m32n8k16.load.{{[ab].*}}.s8.p
; INT-DAG: m16n16k16.load.{{[ab].*}}.u8.p
; INT-DAG: m8n32k16.load.{{[ab].*}}.u8.p
; INT-DAG: m32n8k16.load.{{[ab].*}}.u8.p
; INT-DAG: m32n8k16.{{load|store}}.{{[cd].*\.s32}}.p
; INT-DAG: m16n16k16.mma.{{.*}}.u8
; INT-DAG: m16n16k16.mma.{{.*}}.s8
; INT-DAG: m8n32k16.mma.{{.*}}.u8
; INT-DAG: m8n32k16.mma.{{.*}}.s8
; INT-DAG: m32n8k16.mma.{{.*}}.u8
; INT-DAG: m32n8k16.mma.{{.*}}.s8

; SUBINT-DAG: m8n8k128.load.{{[ab].*}}.b1.p
; SUBINT-DAG: m8n8k32.load.{{[ab].*}}.s4.p
; SUBINT-DAG: m8n8k32.load.{{[ab].*}}.u4.p
; SUBINT-DAG: m8n8k128.{{load|store}}.{{[cd].*\.s32}}.p
; SUBINT-DAG: m8n8k32.{{load|store}}.{{[cd].*\.s32}}.p
; SUBINT-DAG: m8n8k32.mma.{{.*}}.u4
; SUBINT-DAG: m8n8k32.mma.{{.*}}.s4
; SUBINT-DAG: m8n8k128.mma.{{.*}}.b1

; ALTFLOAT-DAG: m16n16k16.load.{{[ab].*}}.bf16.p
; ALTFLOAT-DAG: m8n32k16.load.{{[ab].*}}.bf16.p
; ALTFLOAT-DAG: m32n8k16.load.{{[ab].*}}.bf16.p
; ALTFLOAT-DAG: m16n16k8.load.{{[ab].*}}.tf32.p
; ALTFLOAT-DAG: m16n16k16.mma.{{.*}}.bf16
; ALTFLOAT-DAG: m8n32k16.mma.{{.*}}.bf16
; ALTFLOAT-DAG: m32n8k16.mma.{{.*}}.bf16
; ALTFLOAT-DAG: m16n16k8.mma.{{.*}}.tf32

; DOUBLE-DAG: m8n8k4.load.{{[abc].*}}.f64.p
; DOUBLE-DAG: m8n8k4.store.d.{{.*}}.f64.p
; DOUBLE-DAG: m8n8k4.mma.{{.*}}.f64

; MMA-DAG: mma.m8n8k4.{{.*}}.f16.f32
; MMA-DAG: mma.m8n8k4.{{.*}}.f32.f16
; MMA-DAG: mma.m8n8k4.{{.*}}.f16.f16
; MMA-DAG: mma.m8n8k4.{{.*}}.f32.f32

; PTX65MMA-DAG: mma.m16n8k8.row.col.f16.f16
; PTX65MMA-DAG: mma.m16n8k8.row.col.f32.f32
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.u8.u8
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.s8.s8
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.s8.u8
; PTX65MMA-DAG: mma.m8n8k16.row.col{{.*}}.u8.s8
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.u4.u4
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.s4.s4
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.s4.u4
; PTX65MMA-DAG: mma.m8n8k32.row.col{{.*}}.u4.s4

; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.trans.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.trans.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.trans.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x1.trans.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16
; PTX65LDMATRIX-DAG: ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16

; PTX71MMA-DAG: mma.m8n8k4.row.col.f64
; PTX71MMA-DAG: mma.m16n8k4.row.col.tf32
; PTX71MMA-DAG: mma.m16n8k8.row.col.tf32
; PTX71MMA-DAG: mma.m16n8k16.row.col.bf16
; PTX71MMA-DAG: mma.m16n8k8.row.col.bf16
; PTX71MMA-DAG: mma.m16n8k16.row.col.f16.f16
; PTX71MMA-DAG: mma.m16n8k16.row.col.f32.f32
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.u8.u8
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.s8.s8
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.s8.u8
; PTX71MMA-DAG: mma.m16n8k16.row.col{{.*}}.u8.s8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u8.u8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s8.s8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s8.u8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u8.s8
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u4.u4
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s4.s4
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.s4.u4
; PTX71MMA-DAG: mma.m16n8k32.row.col{{.*}}.u4.s4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.u4.u4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.s4.s4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.s4.u4
; PTX71MMA-DAG: mma.m16n8k64.row.col{{.*}}.u4.s4
; PTX71MMA-DAG: mma.and.popc.m8n8k128.row.col.b1
; PTX71MMA-DAG: mma.xor.popc.m8n8k128.row.col.b1
; PTX71MMA-DAG: mma.and.popc.m16n8k128.row.col.b1
; PTX71MMA-DAG: mma.xor.popc.m16n8k128.row.col.b1
; PTX71MMA-DAG: mma.and.popc.m16n8k256.row.col.b1
; PTX71MMA-DAG: mma.xor.popc.m16n8k256.row.col.b1
;

""")

  print("; INTRINSICS_LIST_BEGIN")
  for intrinsic, instruction in sorted(items):
    print("; ", intrinsic, " -> ", instruction,"")
  print("; INTRINSICS_LIST_END")
  print("; INTRINSICS: ; INTRINSICS_LIST_END")

def gen_tests():
  items = gen_m_load_tests()
  items += gen_m_store_tests()
  items += gen_mmul_tests()
  gen_check_unsupported_ops(items)

parser = argparse.ArgumentParser()
args = parser.parse_args()

gen_tests()
