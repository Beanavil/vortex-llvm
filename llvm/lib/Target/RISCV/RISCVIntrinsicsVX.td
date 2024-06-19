//===- RISCVIntrinsicsVX.td - Vortex Intrinsics Instructions -------*- tblgen -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

let hasSideEffects = 1 in {
// MLOAD
// class RVInstI<bits<3> funct3, RISCVOpcode opcode, dag outs,
//               dag ins, string opcodestr, string argstr>
let mayLoad = 1 in {
def VX_MLOAD_A_2 : RVInstI<0, RISCV_CUSTOM2, (outs TGPR2:$rd),
                           (ins GPR:$rs1, simm12:$imm12),
                           "vx_mload_a_2m2n2k($rd, $rs1, ${imm12})", "">, Sched<[]>;
def VX_MLOAD_B_2 : RVInstI<1, RISCV_CUSTOM2, (outs TGPR2:$rd),
                           (ins GPR:$rs1, simm12:$imm12),
                           "vx_mload_b_2m2n2k($rd, $rs1, ${imm12})", "">, Sched<[]>;
def VX_MLOAD_C_2 : RVInstI<2, RISCV_CUSTOM2, (outs GPR:$rd),
                           (ins GPR:$rs1, simm12:$imm12),
                           "vx_mload_c_2m2n2k($rd, $rs1, ${imm12})", "">, Sched<[]>;
}

// MSTORE
// class RVInstS<bits<3> funct3, RISCVOpcode opcode, dag outs,
//               dag ins, string opcodestr, string argstr>
let mayStore = 1 in {
def VX_MSTORE_2 : RVInstS<5, RISCV_CUSTOM2, (outs),
                          (ins GPR:$rs2, GPR:$rs1, simm12:$imm12),
                          "vx_mstore_d_2m2n2k($rs2, $rs1, ${imm12})", "">, Sched<[]>;
}
}

// MMUL
// class RVInstR<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode, dag outs,
//               dag ins, string opcodestr, string argstr>
def VX_MMUL_2 : RVInstR<0, 3, RISCV_CUSTOM2, (outs GPR:$rd),
                        (ins TGPR2:$rs1, TGPR2:$rs2),
                        "vx_mmul_2m2n2k(${rd}, ${rs1}, ${rs2})", "">, Sched<[]>;

// MADD
// class RVInstR<bits<7> funct7, bits<3> funct3, RISCVOpcode opcode, dag outs,
//               dag ins, string opcodestr, string argstr>
def VX_MADD_2 : RVInstR<0, 4, RISCV_CUSTOM2, (outs GPR:$rd),
                        (ins GPR:$rs1, GPR:$rs2),
                        "vx_madd_2m2n2k(${rd}, ${rs1}, ${rs2})", "">, Sched<[]>;




// MLOAD
// A
def : Pat<(int_riscv_vx_mload_a_m2n2k2 GPR:$rs1, simm12:$imm12),
            (VX_MLOAD_A_2 GPR:$rs1, simm12:$imm12)>;

// B
def : Pat<(int_riscv_vx_mload_b_m2n2k2 GPR:$rs1, simm12:$imm12),
            (VX_MLOAD_B_2 GPR:$rs1, simm12:$imm12)>;

// C
def : Pat<(int_riscv_vx_mload_c_m2n2k2 GPR:$rs1, simm12:$imm12),
            (VX_MLOAD_C_2 GPR:$rs1, simm12:$imm12)>;

// MSTORE
// C
def : Pat<(int_riscv_vx_mstore_d_m2n2k2 GPR:$rs2, GPR:$rs1, simm12:$imm12),
            (VX_MSTORE_2 GPR:$rs2, GPR:$rs1, simm12:$imm12)>;

// MMUL
def : Pat<(int_riscv_vx_mmul_m2n2k2 TGPR2:$rs1, TGPR2:$rs2), (VX_MMUL_2 TGPR2:$rs1, TGPR2:$rs2)>;

// MADD
def : Pat<(int_riscv_vx_madd_m2n2k2 GPR:$rs1, GPR:$rs2), (VX_MADD_2 GPR:$rs1, GPR:$rs2)>;