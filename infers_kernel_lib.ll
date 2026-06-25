; ModuleID = 'builtin.module'
source_filename = "infers_kernel_lib"
target datalayout = "e-p:64:64:64-p3:32:32:32-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64-i128:128:128-f32:32:32-f64:64:64-f128:128:128-v16:16:16-v32:32:32-v64:64:64-v128:128:128-n16:32:64-a:8:8"
target triple = "nvptx64-nvidia-cuda"

@__dynamic_smem_infers_rms_norm_gated_bf16 = external addrspace(3) global [0 x i8], align 16
@__dynamic_smem_infers_l2norm_bf16 = external addrspace(3) global [0 x i8], align 16
@__dynamic_smem_infers_rmsnorm_bf16 = external addrspace(3) global [0 x i8], align 16
@__shared_mem_1 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__shared_mem_0 = addrspace(3) global [256 x float] zeroinitializer, align 4
@__dynamic_smem_infers_softmax_bf16 = external addrspace(3) global [0 x i8], align 16
@__dynamic_smem_infers_paged_attention_decode_bf16 = external addrspace(3) global [0 x i8], align 16
@__dynamic_smem_infers_gdn_update_bf16 = external addrspace(3) global [0 x i8], align 16
@__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16 = external addrspace(3) global [0 x i8], align 16
@__dynamic_smem_int4_gemm_auto_round_tiled = external addrspace(3) global [0 x i8], align 16
@__dynamic_smem_int4_gemm_v3_ksplit_sm = external addrspace(3) global [0 x i8], align 16
define void @int4_gemm_auto_round(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v7, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v9, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v16, %entry ]
  %v26 = phi { ptr, i64 } [ %v18, %entry ]
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v25, ptr %v35, align 8
  %v36 = bitcast i32 %v30 to i32
  %v37 = bitcast i32 %v31 to i32
  %v38 = bitcast i32 %v32 to i32
  %v39 = bitcast i32 %v33 to i32
  %v40 = bitcast i32 %v34 to i32
  %v41 = extractvalue { ptr, i64 } %v26, 0
  %v42 = extractvalue { ptr, i64 } %v26, 1
  %v43 = extractvalue { ptr, i64 } %v27, 0
  %v44 = extractvalue { ptr, i64 } %v27, 1
  %v45 = extractvalue { ptr, i64 } %v28, 0
  %v46 = extractvalue { ptr, i64 } %v28, 1
  %v47 = extractvalue { ptr, i64 } %v29, 0
  %v48 = extractvalue { ptr, i64 } %v29, 1
  call void @int4_gemm_innerNtB2_9AutoRoundEB4_(ptr %v35, ptr %v41, i64 %v42, ptr %v43, i64 %v44, ptr %v45, i64 %v46, ptr %v47, i64 %v48, i32 %v36, i32 %v37, i32 %v38, i32 %v39, i32 %v40) #0
  br label %bb1
bb1:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.y()

define void @int4_gemm_v4_ksplit(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v7, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v9, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v16, %entry ]
  %v26 = phi { ptr, i64 } [ %v18, %entry ]
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v36 = alloca [4 x i32], align 4
  %v37 = alloca [8 x i16], align 2
  %v38 = alloca [4 x i32], align 4
  %v39 = alloca [8 x i16], align 2
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v42 = mul i32 %v41, 64
  %v43 = zext i32 %v42 to i64
  %v44 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v45 = zext i32 %v44 to i64
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb4
bb4:
  %v47 = zext i32 %v46 to i64
  %v48 = zext i32 %v30 to i64
  %v49 = zext i32 %v31 to i64
  %v50 = zext i32 %v32 to i64
  %v51 = zext i32 %v34 to i64
  %v52 = icmp eq i64 %v50, 0
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb5, label %bb60
bb5:
  %v54 = udiv i64 %v49, %v50
  %v55 = mul i64 %v45, 4
  %v56 = add i64 %v43, %v55
  %v57 = icmp uge i64 %v56, %v48
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb7, label %bb6
bb6:
  br label %bb42
bb7:
  %v59 = add i64 %v54, %v51
  %v60 = sub i64 %v59, 1
  %v61 = icmp eq i64 %v51, 0
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb8, label %bb61
bb8:
  %v63 = udiv i64 %v60, %v51
  %v64 = mul i64 %v47, %v63
  %v65 = icmp uge i64 %v64, %v54
  %v66 = xor i1 %v65, 1
  br i1 %v66, label %bb17, label %bb9
bb9:
  br label %bb10
bb10:
  %v67 = phi i64 [ 0, %bb9 ], [ %v885, %bb16 ]
  %v68 = icmp ult i64 %v67, 4
  %v69 = xor i1 %v68, 1
  br i1 %v69, label %bb44, label %bb43
bb11:
  unreachable
bb12:
  %v70 = extractvalue { i64, i64 } %v884, 1
  %v71 = add i64 %v56, %v70
  %v72 = icmp ult i64 %v71, %v48
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb16, label %bb14
bb13:
  br label %bb42
bb14:
  %v74 = mul i64 %v47, %v48
  %v75 = add i64 %v74, %v71
  %v76 = extractvalue { ptr, i64 } %v25, 1
  %v77 = icmp ult i64 %v75, %v76
  br i1 %v77, label %bb15, label %bb62
bb15:
  %v78 = extractvalue { ptr, i64 } %v25, 0
  %v79 = getelementptr inbounds float, ptr %v78, i64 %v75
  store float 0.0, ptr %v79, align 4
  br label %bb16
bb16:
  br label %bb10
bb17:
  %v80 = add i64 %v64, %v63
  %v81 = icmp ugt i64 %v80, %v54
  %v82 = xor i1 %v81, 1
  br i1 %v82, label %bb19, label %bb18
bb18:
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v83 = phi i64 [ %v54, %bb18 ], [ %v80, %bb19 ]
  br label %bb21
bb21:
  %v84 = phi float [ 0.0, %bb20 ], [ %v107, %bb26 ]
  %v85 = phi float [ 0.0, %bb20 ], [ %v108, %bb26 ]
  %v86 = phi float [ 0.0, %bb20 ], [ %v109, %bb26 ]
  %v87 = phi float [ 0.0, %bb20 ], [ %v110, %bb26 ]
  %v88 = phi float [ 0.0, %bb20 ], [ %v111, %bb26 ]
  %v89 = phi float [ 0.0, %bb20 ], [ %v112, %bb26 ]
  %v90 = phi float [ 0.0, %bb20 ], [ %v113, %bb26 ]
  %v91 = phi float [ 0.0, %bb20 ], [ %v114, %bb26 ]
  %v92 = phi i64 [ %v64, %bb20 ], [ %v895, %bb26 ]
  %v93 = icmp ult i64 %v92, %v83
  %v94 = xor i1 %v93, 1
  br i1 %v94, label %bb48, label %bb47
bb22:
  %v95 = extractvalue { i64, i64 } %v894, 1
  %v96 = mul i64 %v95, %v50
  %v97 = mul i64 %v95, %v48
  %v98 = add i64 %v97, %v56
  %v99 = extractvalue { ptr, i64 } %v27, 1
  %v100 = icmp ult i64 %v98, %v99
  %v101 = extractvalue { ptr, i64 } %v27, 0
  %v102 = getelementptr inbounds i16, ptr %v101, i64 %v98
  %v103 = load i16, ptr %v102, align 2
  %v104 = call float @f16_to_f32(i16 %v103) #0
  br label %bb51
bb23:
  %v105 = icmp ult i64 %v56, %v48
  %v106 = xor i1 %v105, 1
  br i1 %v106, label %bb32, label %bb30
bb24:
  %v107 = phi float [ %v835, %bb29 ], [ %v84, %bb55 ]
  %v108 = phi float [ %v836, %bb29 ], [ %v85, %bb55 ]
  %v109 = phi float [ %v837, %bb29 ], [ %v86, %bb55 ]
  %v110 = phi float [ %v838, %bb29 ], [ %v87, %bb55 ]
  %v111 = phi float [ %v839, %bb29 ], [ %v88, %bb55 ]
  %v112 = phi float [ %v840, %bb29 ], [ %v89, %bb55 ]
  %v113 = phi float [ %v841, %bb29 ], [ %v90, %bb55 ]
  %v114 = phi float [ %v842, %bb29 ], [ %v91, %bb55 ]
  %v115 = phi i64 [ %v1011, %bb29 ], [ %v997, %bb55 ]
  %v116 = phi i64 [ %v1012, %bb29 ], [ %v1000, %bb55 ]
  %v117 = add i64 %v1002, 1
  %v118 = icmp eq i64 %v117, 0
  %v119 = select i1 %v118, i8 0, i8 1
  %v120 = insertvalue { i8, { { i64 } } } undef, i8 %v119, 0
  %v121 = insertvalue { i8, { { i64 } } } %v120, i64 %v117, 1, 0, 0
  %v122 = extractvalue { i8, { { i64 } } } %v121, 0
  %v123 = zext i8 %v122 to i64
  %v124 = icmp eq i64 %v123, 1
  %v125 = extractvalue { i8, { { i64 } } } %v121, 1
  %v126 = alloca { { i64 } }, align 8
  store { { i64 } } %v125, ptr %v126, align 8
  %v127 = load i64, ptr %v126, align 8
  %v128 = icmp ugt i64 %v116, 0
  %v129 = xor i1 %v128, 1
  br i1 %v129, label %bb57, label %bb56
bb25:
  %v130 = extractvalue { i64, i64 } %v1010, 1
  %v131 = mul i64 %v130, 8
  %v132 = add i64 %v96, %v131
  %v133 = extractvalue { ptr, i64 } %v26, 0
  %v134 = zext i32 3 to i64
  %v135 = and i64 %v134, 63
  %v136 = lshr i64 %v132, %v135
  %v137 = mul i64 %v136, %v48
  %v138 = add i64 %v137, %v56
  %v139 = getelementptr inbounds i32, ptr %v133, i64 %v138
  %v140 = bitcast ptr %v139 to ptr
  %v141 = load [4 x i32], ptr %v140, align 4
  store [4 x i32] %v141, ptr %v36, align 4
  %v142 = extractvalue { ptr, i64 } %v29, 0
  %v143 = getelementptr inbounds i16, ptr %v142, i64 %v132
  %v144 = bitcast ptr %v143 to ptr
  %v145 = load [8 x i16], ptr %v144, align 2
  store [8 x i16] %v145, ptr %v37, align 2
  %v146 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 0
  %v147 = load i16, ptr %v146, align 2
  %v148 = zext i16 %v147 to i32
  %v149 = and i32 16, 31
  %v150 = shl i32 %v148, %v149
  %v151 = bitcast i32 %v150 to float
  %v152 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 1
  %v153 = load i16, ptr %v152, align 2
  %v154 = zext i16 %v153 to i32
  %v155 = and i32 16, 31
  %v156 = shl i32 %v154, %v155
  %v157 = bitcast i32 %v156 to float
  %v158 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 2
  %v159 = load i16, ptr %v158, align 2
  %v160 = zext i16 %v159 to i32
  %v161 = and i32 16, 31
  %v162 = shl i32 %v160, %v161
  %v163 = bitcast i32 %v162 to float
  %v164 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 3
  %v165 = load i16, ptr %v164, align 2
  %v166 = zext i16 %v165 to i32
  %v167 = and i32 16, 31
  %v168 = shl i32 %v166, %v167
  %v169 = bitcast i32 %v168 to float
  %v170 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 4
  %v171 = load i16, ptr %v170, align 2
  %v172 = zext i16 %v171 to i32
  %v173 = and i32 16, 31
  %v174 = shl i32 %v172, %v173
  %v175 = bitcast i32 %v174 to float
  %v176 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 5
  %v177 = load i16, ptr %v176, align 2
  %v178 = zext i16 %v177 to i32
  %v179 = and i32 16, 31
  %v180 = shl i32 %v178, %v179
  %v181 = bitcast i32 %v180 to float
  %v182 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 6
  %v183 = load i16, ptr %v182, align 2
  %v184 = zext i16 %v183 to i32
  %v185 = and i32 16, 31
  %v186 = shl i32 %v184, %v185
  %v187 = bitcast i32 %v186 to float
  %v188 = getelementptr inbounds [8 x i16], ptr %v37, i32 0, i64 7
  %v189 = load i16, ptr %v188, align 2
  %v190 = zext i16 %v189 to i32
  %v191 = and i32 16, 31
  %v192 = shl i32 %v190, %v191
  %v193 = bitcast i32 %v192 to float
  %v194 = getelementptr inbounds [4 x i32], ptr %v36, i32 0, i64 0
  %v195 = load i32, ptr %v194, align 4
  %v196 = and i32 %v195, 15
  %v197 = trunc i32 %v196 to i8
  %v198 = and i32 4, 31
  %v199 = lshr i32 %v195, %v198
  %v200 = and i32 %v199, 15
  %v201 = trunc i32 %v200 to i8
  %v202 = and i32 8, 31
  %v203 = lshr i32 %v195, %v202
  %v204 = and i32 %v203, 15
  %v205 = trunc i32 %v204 to i8
  %v206 = and i32 12, 31
  %v207 = lshr i32 %v195, %v206
  %v208 = and i32 %v207, 15
  %v209 = trunc i32 %v208 to i8
  %v210 = sitofp i8 %v197 to float
  %v211 = fmul contract float %v210, %v104
  %v212 = fsub contract float %v211, %v979
  %v213 = fmul contract float %v212, %v151
  %v214 = fadd contract float %v107, %v213
  %v215 = sitofp i8 %v201 to float
  %v216 = fmul contract float %v215, %v104
  %v217 = fsub contract float %v216, %v979
  %v218 = fmul contract float %v217, %v157
  %v219 = fadd contract float %v108, %v218
  %v220 = sitofp i8 %v205 to float
  %v221 = fmul contract float %v220, %v104
  %v222 = fsub contract float %v221, %v979
  %v223 = fmul contract float %v222, %v163
  %v224 = fadd contract float %v214, %v223
  %v225 = sitofp i8 %v209 to float
  %v226 = fmul contract float %v225, %v104
  %v227 = fsub contract float %v226, %v979
  %v228 = fmul contract float %v227, %v169
  %v229 = fadd contract float %v219, %v228
  %v230 = and i32 16, 31
  %v231 = lshr i32 %v195, %v230
  %v232 = and i32 %v231, 15
  %v233 = trunc i32 %v232 to i8
  %v234 = and i32 20, 31
  %v235 = lshr i32 %v195, %v234
  %v236 = and i32 %v235, 15
  %v237 = trunc i32 %v236 to i8
  %v238 = and i32 24, 31
  %v239 = lshr i32 %v195, %v238
  %v240 = and i32 %v239, 15
  %v241 = trunc i32 %v240 to i8
  %v242 = and i32 28, 31
  %v243 = lshr i32 %v195, %v242
  %v244 = and i32 %v243, 15
  %v245 = trunc i32 %v244 to i8
  %v246 = sitofp i8 %v233 to float
  %v247 = fmul contract float %v246, %v104
  %v248 = fsub contract float %v247, %v979
  %v249 = fmul contract float %v248, %v175
  %v250 = fadd contract float %v224, %v249
  %v251 = sitofp i8 %v237 to float
  %v252 = fmul contract float %v251, %v104
  %v253 = fsub contract float %v252, %v979
  %v254 = fmul contract float %v253, %v181
  %v255 = fadd contract float %v229, %v254
  %v256 = sitofp i8 %v241 to float
  %v257 = fmul contract float %v256, %v104
  %v258 = fsub contract float %v257, %v979
  %v259 = fmul contract float %v258, %v187
  %v260 = fadd contract float %v250, %v259
  %v261 = sitofp i8 %v245 to float
  %v262 = fmul contract float %v261, %v104
  %v263 = fsub contract float %v262, %v979
  %v264 = fmul contract float %v263, %v193
  %v265 = fadd contract float %v255, %v264
  %v266 = getelementptr inbounds [4 x i32], ptr %v36, i32 0, i64 1
  %v267 = load i32, ptr %v266, align 4
  %v268 = and i32 %v267, 15
  %v269 = trunc i32 %v268 to i8
  %v270 = and i32 4, 31
  %v271 = lshr i32 %v267, %v270
  %v272 = and i32 %v271, 15
  %v273 = trunc i32 %v272 to i8
  %v274 = and i32 8, 31
  %v275 = lshr i32 %v267, %v274
  %v276 = and i32 %v275, 15
  %v277 = trunc i32 %v276 to i8
  %v278 = and i32 12, 31
  %v279 = lshr i32 %v267, %v278
  %v280 = and i32 %v279, 15
  %v281 = trunc i32 %v280 to i8
  %v282 = sitofp i8 %v269 to float
  %v283 = fmul contract float %v282, %v905
  %v284 = fsub contract float %v283, %v982
  %v285 = fmul contract float %v284, %v151
  %v286 = fadd contract float %v109, %v285
  %v287 = sitofp i8 %v273 to float
  %v288 = fmul contract float %v287, %v905
  %v289 = fsub contract float %v288, %v982
  %v290 = fmul contract float %v289, %v157
  %v291 = fadd contract float %v110, %v290
  %v292 = sitofp i8 %v277 to float
  %v293 = fmul contract float %v292, %v905
  %v294 = fsub contract float %v293, %v982
  %v295 = fmul contract float %v294, %v163
  %v296 = fadd contract float %v286, %v295
  %v297 = sitofp i8 %v281 to float
  %v298 = fmul contract float %v297, %v905
  %v299 = fsub contract float %v298, %v982
  %v300 = fmul contract float %v299, %v169
  %v301 = fadd contract float %v291, %v300
  %v302 = and i32 16, 31
  %v303 = lshr i32 %v267, %v302
  %v304 = and i32 %v303, 15
  %v305 = trunc i32 %v304 to i8
  %v306 = and i32 20, 31
  %v307 = lshr i32 %v267, %v306
  %v308 = and i32 %v307, 15
  %v309 = trunc i32 %v308 to i8
  %v310 = and i32 24, 31
  %v311 = lshr i32 %v267, %v310
  %v312 = and i32 %v311, 15
  %v313 = trunc i32 %v312 to i8
  %v314 = and i32 28, 31
  %v315 = lshr i32 %v267, %v314
  %v316 = and i32 %v315, 15
  %v317 = trunc i32 %v316 to i8
  %v318 = sitofp i8 %v305 to float
  %v319 = fmul contract float %v318, %v905
  %v320 = fsub contract float %v319, %v982
  %v321 = fmul contract float %v320, %v175
  %v322 = fadd contract float %v296, %v321
  %v323 = sitofp i8 %v309 to float
  %v324 = fmul contract float %v323, %v905
  %v325 = fsub contract float %v324, %v982
  %v326 = fmul contract float %v325, %v181
  %v327 = fadd contract float %v301, %v326
  %v328 = sitofp i8 %v313 to float
  %v329 = fmul contract float %v328, %v905
  %v330 = fsub contract float %v329, %v982
  %v331 = fmul contract float %v330, %v187
  %v332 = fadd contract float %v322, %v331
  %v333 = sitofp i8 %v317 to float
  %v334 = fmul contract float %v333, %v905
  %v335 = fsub contract float %v334, %v982
  %v336 = fmul contract float %v335, %v193
  %v337 = fadd contract float %v327, %v336
  %v338 = getelementptr inbounds [4 x i32], ptr %v36, i32 0, i64 2
  %v339 = load i32, ptr %v338, align 4
  %v340 = and i32 %v339, 15
  %v341 = trunc i32 %v340 to i8
  %v342 = and i32 4, 31
  %v343 = lshr i32 %v339, %v342
  %v344 = and i32 %v343, 15
  %v345 = trunc i32 %v344 to i8
  %v346 = and i32 8, 31
  %v347 = lshr i32 %v339, %v346
  %v348 = and i32 %v347, 15
  %v349 = trunc i32 %v348 to i8
  %v350 = and i32 12, 31
  %v351 = lshr i32 %v339, %v350
  %v352 = and i32 %v351, 15
  %v353 = trunc i32 %v352 to i8
  %v354 = sitofp i8 %v341 to float
  %v355 = fmul contract float %v354, %v911
  %v356 = fsub contract float %v355, %v985
  %v357 = fmul contract float %v356, %v151
  %v358 = fadd contract float %v111, %v357
  %v359 = sitofp i8 %v345 to float
  %v360 = fmul contract float %v359, %v911
  %v361 = fsub contract float %v360, %v985
  %v362 = fmul contract float %v361, %v157
  %v363 = fadd contract float %v112, %v362
  %v364 = sitofp i8 %v349 to float
  %v365 = fmul contract float %v364, %v911
  %v366 = fsub contract float %v365, %v985
  %v367 = fmul contract float %v366, %v163
  %v368 = fadd contract float %v358, %v367
  %v369 = sitofp i8 %v353 to float
  %v370 = fmul contract float %v369, %v911
  %v371 = fsub contract float %v370, %v985
  %v372 = fmul contract float %v371, %v169
  %v373 = fadd contract float %v363, %v372
  %v374 = and i32 16, 31
  %v375 = lshr i32 %v339, %v374
  %v376 = and i32 %v375, 15
  %v377 = trunc i32 %v376 to i8
  %v378 = and i32 20, 31
  %v379 = lshr i32 %v339, %v378
  %v380 = and i32 %v379, 15
  %v381 = trunc i32 %v380 to i8
  %v382 = and i32 24, 31
  %v383 = lshr i32 %v339, %v382
  %v384 = and i32 %v383, 15
  %v385 = trunc i32 %v384 to i8
  %v386 = and i32 28, 31
  %v387 = lshr i32 %v339, %v386
  %v388 = and i32 %v387, 15
  %v389 = trunc i32 %v388 to i8
  %v390 = sitofp i8 %v377 to float
  %v391 = fmul contract float %v390, %v911
  %v392 = fsub contract float %v391, %v985
  %v393 = fmul contract float %v392, %v175
  %v394 = fadd contract float %v368, %v393
  %v395 = sitofp i8 %v381 to float
  %v396 = fmul contract float %v395, %v911
  %v397 = fsub contract float %v396, %v985
  %v398 = fmul contract float %v397, %v181
  %v399 = fadd contract float %v373, %v398
  %v400 = sitofp i8 %v385 to float
  %v401 = fmul contract float %v400, %v911
  %v402 = fsub contract float %v401, %v985
  %v403 = fmul contract float %v402, %v187
  %v404 = fadd contract float %v394, %v403
  %v405 = sitofp i8 %v389 to float
  %v406 = fmul contract float %v405, %v911
  %v407 = fsub contract float %v406, %v985
  %v408 = fmul contract float %v407, %v193
  %v409 = fadd contract float %v399, %v408
  %v410 = getelementptr inbounds [4 x i32], ptr %v36, i32 0, i64 3
  %v411 = load i32, ptr %v410, align 4
  %v412 = and i32 %v411, 15
  %v413 = trunc i32 %v412 to i8
  %v414 = and i32 4, 31
  %v415 = lshr i32 %v411, %v414
  %v416 = and i32 %v415, 15
  %v417 = trunc i32 %v416 to i8
  %v418 = and i32 8, 31
  %v419 = lshr i32 %v411, %v418
  %v420 = and i32 %v419, 15
  %v421 = trunc i32 %v420 to i8
  %v422 = and i32 12, 31
  %v423 = lshr i32 %v411, %v422
  %v424 = and i32 %v423, 15
  %v425 = trunc i32 %v424 to i8
  %v426 = sitofp i8 %v413 to float
  %v427 = fmul contract float %v426, %v917
  %v428 = fsub contract float %v427, %v988
  %v429 = fmul contract float %v428, %v151
  %v430 = fadd contract float %v113, %v429
  %v431 = sitofp i8 %v417 to float
  %v432 = fmul contract float %v431, %v917
  %v433 = fsub contract float %v432, %v988
  %v434 = fmul contract float %v433, %v157
  %v435 = fadd contract float %v114, %v434
  %v436 = sitofp i8 %v421 to float
  %v437 = fmul contract float %v436, %v917
  %v438 = fsub contract float %v437, %v988
  %v439 = fmul contract float %v438, %v163
  %v440 = fadd contract float %v430, %v439
  %v441 = sitofp i8 %v425 to float
  %v442 = fmul contract float %v441, %v917
  %v443 = fsub contract float %v442, %v988
  %v444 = fmul contract float %v443, %v169
  %v445 = fadd contract float %v435, %v444
  %v446 = and i32 16, 31
  %v447 = lshr i32 %v411, %v446
  %v448 = and i32 %v447, 15
  %v449 = trunc i32 %v448 to i8
  %v450 = and i32 20, 31
  %v451 = lshr i32 %v411, %v450
  %v452 = and i32 %v451, 15
  %v453 = trunc i32 %v452 to i8
  %v454 = and i32 24, 31
  %v455 = lshr i32 %v411, %v454
  %v456 = and i32 %v455, 15
  %v457 = trunc i32 %v456 to i8
  %v458 = and i32 28, 31
  %v459 = lshr i32 %v411, %v458
  %v460 = and i32 %v459, 15
  %v461 = trunc i32 %v460 to i8
  %v462 = sitofp i8 %v449 to float
  %v463 = fmul contract float %v462, %v917
  %v464 = fsub contract float %v463, %v988
  %v465 = fmul contract float %v464, %v175
  %v466 = fadd contract float %v440, %v465
  %v467 = sitofp i8 %v453 to float
  %v468 = fmul contract float %v467, %v917
  %v469 = fsub contract float %v468, %v988
  %v470 = fmul contract float %v469, %v181
  %v471 = fadd contract float %v445, %v470
  %v472 = sitofp i8 %v457 to float
  %v473 = fmul contract float %v472, %v917
  %v474 = fsub contract float %v473, %v988
  %v475 = fmul contract float %v474, %v187
  %v476 = fadd contract float %v466, %v475
  %v477 = sitofp i8 %v461 to float
  %v478 = fmul contract float %v477, %v917
  %v479 = fsub contract float %v478, %v988
  %v480 = fmul contract float %v479, %v193
  %v481 = fadd contract float %v471, %v480
  %v482 = add i64 %v130, 1
  %v483 = icmp ult i64 %v482, %v989
  %v484 = xor i1 %v483, 1
  br i1 %v484, label %bb28, label %bb27
bb26:
  br label %bb21
bb27:
  %v485 = add i64 %v132, 8
  %v486 = extractvalue { ptr, i64 } %v26, 0
  %v487 = zext i32 3 to i64
  %v488 = and i64 %v487, 63
  %v489 = lshr i64 %v485, %v488
  %v490 = mul i64 %v489, %v48
  %v491 = add i64 %v490, %v56
  %v492 = getelementptr inbounds i32, ptr %v486, i64 %v491
  %v493 = bitcast ptr %v492 to ptr
  %v494 = load [4 x i32], ptr %v493, align 4
  store [4 x i32] %v494, ptr %v38, align 4
  %v495 = extractvalue { ptr, i64 } %v29, 0
  %v496 = getelementptr inbounds i16, ptr %v495, i64 %v485
  %v497 = bitcast ptr %v496 to ptr
  %v498 = load [8 x i16], ptr %v497, align 2
  store [8 x i16] %v498, ptr %v39, align 2
  %v499 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 0
  %v500 = load i16, ptr %v499, align 2
  %v501 = zext i16 %v500 to i32
  %v502 = and i32 16, 31
  %v503 = shl i32 %v501, %v502
  %v504 = bitcast i32 %v503 to float
  %v505 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 1
  %v506 = load i16, ptr %v505, align 2
  %v507 = zext i16 %v506 to i32
  %v508 = and i32 16, 31
  %v509 = shl i32 %v507, %v508
  %v510 = bitcast i32 %v509 to float
  %v511 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 2
  %v512 = load i16, ptr %v511, align 2
  %v513 = zext i16 %v512 to i32
  %v514 = and i32 16, 31
  %v515 = shl i32 %v513, %v514
  %v516 = bitcast i32 %v515 to float
  %v517 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 3
  %v518 = load i16, ptr %v517, align 2
  %v519 = zext i16 %v518 to i32
  %v520 = and i32 16, 31
  %v521 = shl i32 %v519, %v520
  %v522 = bitcast i32 %v521 to float
  %v523 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 4
  %v524 = load i16, ptr %v523, align 2
  %v525 = zext i16 %v524 to i32
  %v526 = and i32 16, 31
  %v527 = shl i32 %v525, %v526
  %v528 = bitcast i32 %v527 to float
  %v529 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 5
  %v530 = load i16, ptr %v529, align 2
  %v531 = zext i16 %v530 to i32
  %v532 = and i32 16, 31
  %v533 = shl i32 %v531, %v532
  %v534 = bitcast i32 %v533 to float
  %v535 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 6
  %v536 = load i16, ptr %v535, align 2
  %v537 = zext i16 %v536 to i32
  %v538 = and i32 16, 31
  %v539 = shl i32 %v537, %v538
  %v540 = bitcast i32 %v539 to float
  %v541 = getelementptr inbounds [8 x i16], ptr %v39, i32 0, i64 7
  %v542 = load i16, ptr %v541, align 2
  %v543 = zext i16 %v542 to i32
  %v544 = and i32 16, 31
  %v545 = shl i32 %v543, %v544
  %v546 = bitcast i32 %v545 to float
  %v547 = getelementptr inbounds [4 x i32], ptr %v38, i32 0, i64 0
  %v548 = load i32, ptr %v547, align 4
  %v549 = and i32 %v548, 15
  %v550 = trunc i32 %v549 to i8
  %v551 = and i32 4, 31
  %v552 = lshr i32 %v548, %v551
  %v553 = and i32 %v552, 15
  %v554 = trunc i32 %v553 to i8
  %v555 = and i32 8, 31
  %v556 = lshr i32 %v548, %v555
  %v557 = and i32 %v556, 15
  %v558 = trunc i32 %v557 to i8
  %v559 = and i32 12, 31
  %v560 = lshr i32 %v548, %v559
  %v561 = and i32 %v560, 15
  %v562 = trunc i32 %v561 to i8
  %v563 = sitofp i8 %v550 to float
  %v564 = fmul contract float %v563, %v104
  %v565 = fsub contract float %v564, %v979
  %v566 = fmul contract float %v565, %v504
  %v567 = fadd contract float %v260, %v566
  %v568 = sitofp i8 %v554 to float
  %v569 = fmul contract float %v568, %v104
  %v570 = fsub contract float %v569, %v979
  %v571 = fmul contract float %v570, %v510
  %v572 = fadd contract float %v265, %v571
  %v573 = sitofp i8 %v558 to float
  %v574 = fmul contract float %v573, %v104
  %v575 = fsub contract float %v574, %v979
  %v576 = fmul contract float %v575, %v516
  %v577 = fadd contract float %v567, %v576
  %v578 = sitofp i8 %v562 to float
  %v579 = fmul contract float %v578, %v104
  %v580 = fsub contract float %v579, %v979
  %v581 = fmul contract float %v580, %v522
  %v582 = fadd contract float %v572, %v581
  %v583 = and i32 16, 31
  %v584 = lshr i32 %v548, %v583
  %v585 = and i32 %v584, 15
  %v586 = trunc i32 %v585 to i8
  %v587 = and i32 20, 31
  %v588 = lshr i32 %v548, %v587
  %v589 = and i32 %v588, 15
  %v590 = trunc i32 %v589 to i8
  %v591 = and i32 24, 31
  %v592 = lshr i32 %v548, %v591
  %v593 = and i32 %v592, 15
  %v594 = trunc i32 %v593 to i8
  %v595 = and i32 28, 31
  %v596 = lshr i32 %v548, %v595
  %v597 = and i32 %v596, 15
  %v598 = trunc i32 %v597 to i8
  %v599 = sitofp i8 %v586 to float
  %v600 = fmul contract float %v599, %v104
  %v601 = fsub contract float %v600, %v979
  %v602 = fmul contract float %v601, %v528
  %v603 = fadd contract float %v577, %v602
  %v604 = sitofp i8 %v590 to float
  %v605 = fmul contract float %v604, %v104
  %v606 = fsub contract float %v605, %v979
  %v607 = fmul contract float %v606, %v534
  %v608 = fadd contract float %v582, %v607
  %v609 = sitofp i8 %v594 to float
  %v610 = fmul contract float %v609, %v104
  %v611 = fsub contract float %v610, %v979
  %v612 = fmul contract float %v611, %v540
  %v613 = fadd contract float %v603, %v612
  %v614 = sitofp i8 %v598 to float
  %v615 = fmul contract float %v614, %v104
  %v616 = fsub contract float %v615, %v979
  %v617 = fmul contract float %v616, %v546
  %v618 = fadd contract float %v608, %v617
  %v619 = getelementptr inbounds [4 x i32], ptr %v38, i32 0, i64 1
  %v620 = load i32, ptr %v619, align 4
  %v621 = and i32 %v620, 15
  %v622 = trunc i32 %v621 to i8
  %v623 = and i32 4, 31
  %v624 = lshr i32 %v620, %v623
  %v625 = and i32 %v624, 15
  %v626 = trunc i32 %v625 to i8
  %v627 = and i32 8, 31
  %v628 = lshr i32 %v620, %v627
  %v629 = and i32 %v628, 15
  %v630 = trunc i32 %v629 to i8
  %v631 = and i32 12, 31
  %v632 = lshr i32 %v620, %v631
  %v633 = and i32 %v632, 15
  %v634 = trunc i32 %v633 to i8
  %v635 = sitofp i8 %v622 to float
  %v636 = fmul contract float %v635, %v905
  %v637 = fsub contract float %v636, %v982
  %v638 = fmul contract float %v637, %v504
  %v639 = fadd contract float %v332, %v638
  %v640 = sitofp i8 %v626 to float
  %v641 = fmul contract float %v640, %v905
  %v642 = fsub contract float %v641, %v982
  %v643 = fmul contract float %v642, %v510
  %v644 = fadd contract float %v337, %v643
  %v645 = sitofp i8 %v630 to float
  %v646 = fmul contract float %v645, %v905
  %v647 = fsub contract float %v646, %v982
  %v648 = fmul contract float %v647, %v516
  %v649 = fadd contract float %v639, %v648
  %v650 = sitofp i8 %v634 to float
  %v651 = fmul contract float %v650, %v905
  %v652 = fsub contract float %v651, %v982
  %v653 = fmul contract float %v652, %v522
  %v654 = fadd contract float %v644, %v653
  %v655 = and i32 16, 31
  %v656 = lshr i32 %v620, %v655
  %v657 = and i32 %v656, 15
  %v658 = trunc i32 %v657 to i8
  %v659 = and i32 20, 31
  %v660 = lshr i32 %v620, %v659
  %v661 = and i32 %v660, 15
  %v662 = trunc i32 %v661 to i8
  %v663 = and i32 24, 31
  %v664 = lshr i32 %v620, %v663
  %v665 = and i32 %v664, 15
  %v666 = trunc i32 %v665 to i8
  %v667 = and i32 28, 31
  %v668 = lshr i32 %v620, %v667
  %v669 = and i32 %v668, 15
  %v670 = trunc i32 %v669 to i8
  %v671 = sitofp i8 %v658 to float
  %v672 = fmul contract float %v671, %v905
  %v673 = fsub contract float %v672, %v982
  %v674 = fmul contract float %v673, %v528
  %v675 = fadd contract float %v649, %v674
  %v676 = sitofp i8 %v662 to float
  %v677 = fmul contract float %v676, %v905
  %v678 = fsub contract float %v677, %v982
  %v679 = fmul contract float %v678, %v534
  %v680 = fadd contract float %v654, %v679
  %v681 = sitofp i8 %v666 to float
  %v682 = fmul contract float %v681, %v905
  %v683 = fsub contract float %v682, %v982
  %v684 = fmul contract float %v683, %v540
  %v685 = fadd contract float %v675, %v684
  %v686 = sitofp i8 %v670 to float
  %v687 = fmul contract float %v686, %v905
  %v688 = fsub contract float %v687, %v982
  %v689 = fmul contract float %v688, %v546
  %v690 = fadd contract float %v680, %v689
  %v691 = getelementptr inbounds [4 x i32], ptr %v38, i32 0, i64 2
  %v692 = load i32, ptr %v691, align 4
  %v693 = and i32 %v692, 15
  %v694 = trunc i32 %v693 to i8
  %v695 = and i32 4, 31
  %v696 = lshr i32 %v692, %v695
  %v697 = and i32 %v696, 15
  %v698 = trunc i32 %v697 to i8
  %v699 = and i32 8, 31
  %v700 = lshr i32 %v692, %v699
  %v701 = and i32 %v700, 15
  %v702 = trunc i32 %v701 to i8
  %v703 = and i32 12, 31
  %v704 = lshr i32 %v692, %v703
  %v705 = and i32 %v704, 15
  %v706 = trunc i32 %v705 to i8
  %v707 = sitofp i8 %v694 to float
  %v708 = fmul contract float %v707, %v911
  %v709 = fsub contract float %v708, %v985
  %v710 = fmul contract float %v709, %v504
  %v711 = fadd contract float %v404, %v710
  %v712 = sitofp i8 %v698 to float
  %v713 = fmul contract float %v712, %v911
  %v714 = fsub contract float %v713, %v985
  %v715 = fmul contract float %v714, %v510
  %v716 = fadd contract float %v409, %v715
  %v717 = sitofp i8 %v702 to float
  %v718 = fmul contract float %v717, %v911
  %v719 = fsub contract float %v718, %v985
  %v720 = fmul contract float %v719, %v516
  %v721 = fadd contract float %v711, %v720
  %v722 = sitofp i8 %v706 to float
  %v723 = fmul contract float %v722, %v911
  %v724 = fsub contract float %v723, %v985
  %v725 = fmul contract float %v724, %v522
  %v726 = fadd contract float %v716, %v725
  %v727 = and i32 16, 31
  %v728 = lshr i32 %v692, %v727
  %v729 = and i32 %v728, 15
  %v730 = trunc i32 %v729 to i8
  %v731 = and i32 20, 31
  %v732 = lshr i32 %v692, %v731
  %v733 = and i32 %v732, 15
  %v734 = trunc i32 %v733 to i8
  %v735 = and i32 24, 31
  %v736 = lshr i32 %v692, %v735
  %v737 = and i32 %v736, 15
  %v738 = trunc i32 %v737 to i8
  %v739 = and i32 28, 31
  %v740 = lshr i32 %v692, %v739
  %v741 = and i32 %v740, 15
  %v742 = trunc i32 %v741 to i8
  %v743 = sitofp i8 %v730 to float
  %v744 = fmul contract float %v743, %v911
  %v745 = fsub contract float %v744, %v985
  %v746 = fmul contract float %v745, %v528
  %v747 = fadd contract float %v721, %v746
  %v748 = sitofp i8 %v734 to float
  %v749 = fmul contract float %v748, %v911
  %v750 = fsub contract float %v749, %v985
  %v751 = fmul contract float %v750, %v534
  %v752 = fadd contract float %v726, %v751
  %v753 = sitofp i8 %v738 to float
  %v754 = fmul contract float %v753, %v911
  %v755 = fsub contract float %v754, %v985
  %v756 = fmul contract float %v755, %v540
  %v757 = fadd contract float %v747, %v756
  %v758 = sitofp i8 %v742 to float
  %v759 = fmul contract float %v758, %v911
  %v760 = fsub contract float %v759, %v985
  %v761 = fmul contract float %v760, %v546
  %v762 = fadd contract float %v752, %v761
  %v763 = getelementptr inbounds [4 x i32], ptr %v38, i32 0, i64 3
  %v764 = load i32, ptr %v763, align 4
  %v765 = and i32 %v764, 15
  %v766 = trunc i32 %v765 to i8
  %v767 = and i32 4, 31
  %v768 = lshr i32 %v764, %v767
  %v769 = and i32 %v768, 15
  %v770 = trunc i32 %v769 to i8
  %v771 = and i32 8, 31
  %v772 = lshr i32 %v764, %v771
  %v773 = and i32 %v772, 15
  %v774 = trunc i32 %v773 to i8
  %v775 = and i32 12, 31
  %v776 = lshr i32 %v764, %v775
  %v777 = and i32 %v776, 15
  %v778 = trunc i32 %v777 to i8
  %v779 = sitofp i8 %v766 to float
  %v780 = fmul contract float %v779, %v917
  %v781 = fsub contract float %v780, %v988
  %v782 = fmul contract float %v781, %v504
  %v783 = fadd contract float %v476, %v782
  %v784 = sitofp i8 %v770 to float
  %v785 = fmul contract float %v784, %v917
  %v786 = fsub contract float %v785, %v988
  %v787 = fmul contract float %v786, %v510
  %v788 = fadd contract float %v481, %v787
  %v789 = sitofp i8 %v774 to float
  %v790 = fmul contract float %v789, %v917
  %v791 = fsub contract float %v790, %v988
  %v792 = fmul contract float %v791, %v516
  %v793 = fadd contract float %v783, %v792
  %v794 = sitofp i8 %v778 to float
  %v795 = fmul contract float %v794, %v917
  %v796 = fsub contract float %v795, %v988
  %v797 = fmul contract float %v796, %v522
  %v798 = fadd contract float %v788, %v797
  %v799 = and i32 16, 31
  %v800 = lshr i32 %v764, %v799
  %v801 = and i32 %v800, 15
  %v802 = trunc i32 %v801 to i8
  %v803 = and i32 20, 31
  %v804 = lshr i32 %v764, %v803
  %v805 = and i32 %v804, 15
  %v806 = trunc i32 %v805 to i8
  %v807 = and i32 24, 31
  %v808 = lshr i32 %v764, %v807
  %v809 = and i32 %v808, 15
  %v810 = trunc i32 %v809 to i8
  %v811 = and i32 28, 31
  %v812 = lshr i32 %v764, %v811
  %v813 = and i32 %v812, 15
  %v814 = trunc i32 %v813 to i8
  %v815 = sitofp i8 %v802 to float
  %v816 = fmul contract float %v815, %v917
  %v817 = fsub contract float %v816, %v988
  %v818 = fmul contract float %v817, %v528
  %v819 = fadd contract float %v793, %v818
  %v820 = sitofp i8 %v806 to float
  %v821 = fmul contract float %v820, %v917
  %v822 = fsub contract float %v821, %v988
  %v823 = fmul contract float %v822, %v534
  %v824 = fadd contract float %v798, %v823
  %v825 = sitofp i8 %v810 to float
  %v826 = fmul contract float %v825, %v917
  %v827 = fsub contract float %v826, %v988
  %v828 = fmul contract float %v827, %v540
  %v829 = fadd contract float %v819, %v828
  %v830 = sitofp i8 %v814 to float
  %v831 = fmul contract float %v830, %v917
  %v832 = fsub contract float %v831, %v988
  %v833 = fmul contract float %v832, %v546
  %v834 = fadd contract float %v824, %v833
  br label %bb29
bb28:
  br label %bb29
bb29:
  %v835 = phi float [ %v613, %bb27 ], [ %v260, %bb28 ]
  %v836 = phi float [ %v618, %bb27 ], [ %v265, %bb28 ]
  %v837 = phi float [ %v685, %bb27 ], [ %v332, %bb28 ]
  %v838 = phi float [ %v690, %bb27 ], [ %v337, %bb28 ]
  %v839 = phi float [ %v757, %bb27 ], [ %v404, %bb28 ]
  %v840 = phi float [ %v762, %bb27 ], [ %v409, %bb28 ]
  %v841 = phi float [ %v829, %bb27 ], [ %v476, %bb28 ]
  %v842 = phi float [ %v834, %bb27 ], [ %v481, %bb28 ]
  br label %bb24
bb30:
  %v843 = mul i64 %v47, %v48
  %v844 = add i64 %v843, %v56
  %v845 = extractvalue { ptr, i64 } %v25, 1
  %v846 = icmp ult i64 %v844, %v845
  br i1 %v846, label %bb31, label %bb63
bb31:
  %v847 = fadd contract float %v84, %v85
  %v848 = extractvalue { ptr, i64 } %v25, 0
  %v849 = getelementptr inbounds float, ptr %v848, i64 %v844
  store float %v847, ptr %v849, align 4
  br label %bb32
bb32:
  %v850 = add i64 %v56, 1
  %v851 = icmp ult i64 %v850, %v48
  %v852 = xor i1 %v851, 1
  br i1 %v852, label %bb35, label %bb33
bb33:
  %v853 = mul i64 %v47, %v48
  %v854 = add i64 %v853, %v850
  %v855 = extractvalue { ptr, i64 } %v25, 1
  %v856 = icmp ult i64 %v854, %v855
  br i1 %v856, label %bb34, label %bb64
bb34:
  %v857 = fadd contract float %v86, %v87
  %v858 = extractvalue { ptr, i64 } %v25, 0
  %v859 = getelementptr inbounds float, ptr %v858, i64 %v854
  store float %v857, ptr %v859, align 4
  br label %bb35
bb35:
  %v860 = add i64 %v56, 2
  %v861 = icmp ult i64 %v860, %v48
  %v862 = xor i1 %v861, 1
  br i1 %v862, label %bb38, label %bb36
bb36:
  %v863 = mul i64 %v47, %v48
  %v864 = add i64 %v863, %v860
  %v865 = extractvalue { ptr, i64 } %v25, 1
  %v866 = icmp ult i64 %v864, %v865
  br i1 %v866, label %bb37, label %bb65
bb37:
  %v867 = fadd contract float %v88, %v89
  %v868 = extractvalue { ptr, i64 } %v25, 0
  %v869 = getelementptr inbounds float, ptr %v868, i64 %v864
  store float %v867, ptr %v869, align 4
  br label %bb38
bb38:
  %v870 = add i64 %v56, 3
  %v871 = icmp ult i64 %v870, %v48
  %v872 = xor i1 %v871, 1
  br i1 %v872, label %bb41, label %bb39
bb39:
  %v873 = mul i64 %v47, %v48
  %v874 = add i64 %v873, %v870
  %v875 = extractvalue { ptr, i64 } %v25, 1
  %v876 = icmp ult i64 %v874, %v875
  br i1 %v876, label %bb40, label %bb66
bb40:
  %v877 = fadd contract float %v90, %v91
  %v878 = extractvalue { ptr, i64 } %v25, 0
  %v879 = getelementptr inbounds float, ptr %v878, i64 %v874
  store float %v877, ptr %v879, align 4
  br label %bb41
bb41:
  br label %bb42
bb42:
  ret void
bb43:
  %v880 = add i64 %v67, 1
  %v881 = insertvalue { i64, i64 } undef, i64 1, 0
  %v882 = insertvalue { i64, i64 } %v881, i64 %v67, 1
  br label %bb45
bb44:
  %v883 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb45
bb45:
  %v884 = phi { i64, i64 } [ %v882, %bb43 ], [ %v883, %bb44 ]
  %v885 = phi i64 [ %v880, %bb43 ], [ %v67, %bb44 ]
  %v886 = extractvalue { i64, i64 } %v884, 0
  %v887 = bitcast i64 %v886 to i64
  %v888 = icmp eq i64 %v887, 0
  br i1 %v888, label %bb13, label %bb46
bb46:
  %v889 = icmp eq i64 %v887, 1
  br i1 %v889, label %bb12, label %bb11
bb47:
  %v890 = add i64 %v92, 1
  %v891 = insertvalue { i64, i64 } undef, i64 1, 0
  %v892 = insertvalue { i64, i64 } %v891, i64 %v92, 1
  br label %bb49
bb48:
  %v893 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb49
bb49:
  %v894 = phi { i64, i64 } [ %v892, %bb47 ], [ %v893, %bb48 ]
  %v895 = phi i64 [ %v890, %bb47 ], [ %v92, %bb48 ]
  %v896 = extractvalue { i64, i64 } %v894, 0
  %v897 = bitcast i64 %v896 to i64
  %v898 = icmp eq i64 %v897, 0
  br i1 %v898, label %bb23, label %bb50
bb50:
  %v899 = icmp eq i64 %v897, 1
  br i1 %v899, label %bb22, label %bb11
bb51:
  %v900 = add i64 %v98, 1
  %v901 = icmp ult i64 %v900, %v99
  %v902 = extractvalue { ptr, i64 } %v27, 0
  %v903 = getelementptr inbounds i16, ptr %v902, i64 %v900
  %v904 = load i16, ptr %v903, align 2
  %v905 = call float @f16_to_f32(i16 %v904) #0
  br label %bb52
bb52:
  %v906 = add i64 %v98, 2
  %v907 = icmp ult i64 %v906, %v99
  %v908 = extractvalue { ptr, i64 } %v27, 0
  %v909 = getelementptr inbounds i16, ptr %v908, i64 %v906
  %v910 = load i16, ptr %v909, align 2
  %v911 = call float @f16_to_f32(i16 %v910) #0
  br label %bb53
bb53:
  %v912 = add i64 %v98, 3
  %v913 = icmp ult i64 %v912, %v99
  %v914 = extractvalue { ptr, i64 } %v27, 0
  %v915 = getelementptr inbounds i16, ptr %v914, i64 %v912
  %v916 = load i16, ptr %v915, align 2
  %v917 = call float @f16_to_f32(i16 %v916) #0
  br label %bb54
bb54:
  %v918 = add i64 %v48, 7
  %v919 = udiv i64 %v918, 8
  %v920 = mul i64 %v95, %v919
  %v921 = udiv i64 %v56, 8
  %v922 = add i64 %v920, %v921
  %v923 = extractvalue { ptr, i64 } %v28, 1
  %v924 = icmp ult i64 %v922, %v923
  %v925 = extractvalue { ptr, i64 } %v28, 0
  %v926 = getelementptr inbounds i32, ptr %v925, i64 %v922
  %v927 = load i32, ptr %v926, align 4
  %v928 = urem i64 %v56, 8
  %v929 = mul i64 %v928, 4
  %v930 = trunc i64 %v929 to i32
  %v931 = and i32 %v930, 31
  %v932 = lshr i32 %v927, %v931
  %v933 = and i32 %v932, 15
  %v934 = trunc i32 %v933 to i8
  %v935 = add i64 %v56, 1
  %v936 = udiv i64 %v935, 8
  %v937 = add i64 %v920, %v936
  %v938 = icmp ult i64 %v937, %v923
  %v939 = extractvalue { ptr, i64 } %v28, 0
  %v940 = getelementptr inbounds i32, ptr %v939, i64 %v937
  %v941 = load i32, ptr %v940, align 4
  %v942 = urem i64 %v935, 8
  %v943 = mul i64 %v942, 4
  %v944 = trunc i64 %v943 to i32
  %v945 = and i32 %v944, 31
  %v946 = lshr i32 %v941, %v945
  %v947 = and i32 %v946, 15
  %v948 = trunc i32 %v947 to i8
  %v949 = add i64 %v56, 2
  %v950 = udiv i64 %v949, 8
  %v951 = add i64 %v920, %v950
  %v952 = icmp ult i64 %v951, %v923
  %v953 = extractvalue { ptr, i64 } %v28, 0
  %v954 = getelementptr inbounds i32, ptr %v953, i64 %v951
  %v955 = load i32, ptr %v954, align 4
  %v956 = urem i64 %v949, 8
  %v957 = mul i64 %v956, 4
  %v958 = trunc i64 %v957 to i32
  %v959 = and i32 %v958, 31
  %v960 = lshr i32 %v955, %v959
  %v961 = and i32 %v960, 15
  %v962 = trunc i32 %v961 to i8
  %v963 = add i64 %v56, 3
  %v964 = udiv i64 %v963, 8
  %v965 = add i64 %v920, %v964
  %v966 = icmp ult i64 %v965, %v923
  %v967 = extractvalue { ptr, i64 } %v28, 0
  %v968 = getelementptr inbounds i32, ptr %v967, i64 %v965
  %v969 = load i32, ptr %v968, align 4
  %v970 = urem i64 %v963, 8
  %v971 = mul i64 %v970, 4
  %v972 = trunc i64 %v971 to i32
  %v973 = and i32 %v972, 31
  %v974 = lshr i32 %v969, %v973
  %v975 = and i32 %v974, 15
  %v976 = trunc i32 %v975 to i8
  %v977 = sitofp i8 %v934 to float
  %v978 = fadd contract float %v977, 1.0
  %v979 = fmul contract float %v978, %v104
  %v980 = sitofp i8 %v948 to float
  %v981 = fadd contract float %v980, 1.0
  %v982 = fmul contract float %v981, %v905
  %v983 = sitofp i8 %v962 to float
  %v984 = fadd contract float %v983, 1.0
  %v985 = fmul contract float %v984, %v911
  %v986 = sitofp i8 %v976 to float
  %v987 = fadd contract float %v986, 1.0
  %v988 = fmul contract float %v987, %v917
  %v989 = udiv i64 %v50, 8
  %v990 = insertvalue { i64, i64 } undef, i64 0, 0
  %v991 = insertvalue { i64, i64 } %v990, i64 %v989, 1
  %v992 = extractvalue { i64, i64 } %v991, 0
  %v993 = extractvalue { i64, i64 } %v991, 1
  %v994 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v992, i64 %v993, i64 2) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v994, ptr %v35, align 8
  br label %bb55
bb55:
  %v995 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v996 = getelementptr inbounds { i64, i64 }, ptr %v995, i32 0, i32 0
  %v997 = load i64, ptr %v996, align 8
  %v998 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v999 = getelementptr inbounds { i64, i64 }, ptr %v998, i32 0, i32 1
  %v1000 = load i64, ptr %v999, align 8
  %v1001 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 1
  %v1002 = load i64, ptr %v1001, align 8
  %v1003 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 2
  %v1004 = load i1, ptr %v1003, align 1
  br label %bb24
bb56:
  %v1005 = add i64 %v115, %v127
  %v1006 = sub i64 %v116, 1
  %v1007 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1008 = insertvalue { i64, i64 } %v1007, i64 %v115, 1
  br label %bb58
bb57:
  %v1009 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb58
bb58:
  %v1010 = phi { i64, i64 } [ %v1008, %bb56 ], [ %v1009, %bb57 ]
  %v1011 = phi i64 [ %v1005, %bb56 ], [ %v115, %bb57 ]
  %v1012 = phi i64 [ %v1006, %bb56 ], [ %v116, %bb57 ]
  %v1013 = extractvalue { i64, i64 } %v1010, 0
  %v1014 = bitcast i64 %v1013 to i64
  %v1015 = icmp eq i64 %v1014, 0
  br i1 %v1015, label %bb26, label %bb59
bb59:
  %v1016 = icmp eq i64 %v1014, 1
  br i1 %v1016, label %bb25, label %bb11
bb60:
  unreachable
bb61:
  unreachable
bb62:
  unreachable
bb63:
  unreachable
bb64:
  unreachable
bb65:
  unreachable
bb66:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.y()
declare float @llvm.nvvm.shfl.sync.bfly.f32(i32, float, i32, i32) #0

define void @int4_gemm_warp_split(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v7, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v9, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v16, %entry ]
  %v26 = phi { ptr, i64 } [ %v18, %entry ]
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v37 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb2
bb2:
  %v38 = zext i32 %v37 to i64
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb3
bb3:
  %v40 = zext i32 %v39 to i64
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb4
bb4:
  %v42 = mul i32 %v41, 8
  %v43 = trunc i64 %v40 to i32
  %v44 = add i32 %v42, %v43
  %v45 = zext i32 %v44 to i64
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb5
bb5:
  %v47 = zext i32 %v46 to i64
  %v48 = zext i32 %v30 to i64
  %v49 = zext i32 %v31 to i64
  %v50 = zext i32 %v32 to i64
  %v51 = icmp eq i64 %v50, 0
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb6, label %bb54
bb6:
  %v53 = udiv i64 %v49, %v50
  %v54 = icmp uge i64 %v45, %v48
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb8, label %bb7
bb7:
  br label %bb38
bb8:
  %v56 = zext i32 %v34 to i64
  %v57 = add i64 %v53, %v56
  %v58 = sub i64 %v57, 1
  %v59 = icmp eq i64 %v56, 0
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb9, label %bb55
bb9:
  %v61 = udiv i64 %v58, %v56
  %v62 = mul i64 %v47, %v61
  %v63 = add i64 %v47, 1
  %v64 = mul i64 %v63, %v61
  %v65 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v64, i64 %v53) #0
  br label %bb10
bb10:
  %v66 = add i64 %v62, %v38
  br label %bb11
bb11:
  %v67 = phi float [ 0.0, %bb10 ], [ %v119, %bb25 ]
  %v68 = phi i64 [ %v66, %bb10 ], [ %v138, %bb25 ]
  %v69 = icmp ult i64 %v68, %v65
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb34, label %bb12
bb12:
  %v71 = mul i64 %v68, %v50
  %v72 = icmp ne i32 %v33, 0
  %v73 = icmp eq i32 %v33, 0
  br i1 %v73, label %bb15, label %bb13
bb13:
  %v74 = mul i64 %v68, %v48
  %v75 = add i64 %v74, %v45
  %v76 = extractvalue { ptr, i64 } %v27, 1
  %v77 = icmp ult i64 %v75, %v76
  br i1 %v77, label %bb14, label %bb56
bb14:
  %v78 = extractvalue { ptr, i64 } %v27, 0
  %v79 = getelementptr inbounds i16, ptr %v78, i64 %v75
  %v80 = load i16, ptr %v79, align 2
  br label %bb17
bb15:
  %v81 = mul i64 %v45, %v53
  %v82 = add i64 %v81, %v68
  %v83 = extractvalue { ptr, i64 } %v27, 1
  %v84 = icmp ult i64 %v82, %v83
  br i1 %v84, label %bb16, label %bb57
bb16:
  %v85 = extractvalue { ptr, i64 } %v27, 0
  %v86 = getelementptr inbounds i16, ptr %v85, i64 %v82
  %v87 = load i16, ptr %v86, align 2
  br label %bb17
bb17:
  %v88 = phi i16 [ %v80, %bb14 ], [ %v87, %bb16 ]
  %v89 = call float @f16_to_f32(i16 %v88) #0
  br label %bb39
bb18:
  %v90 = add i64 %v48, 7
  %v91 = udiv i64 %v90, 8
  %v92 = mul i64 %v68, %v91
  %v93 = udiv i64 %v45, 8
  %v94 = add i64 %v92, %v93
  %v95 = urem i64 %v45, 8
  %v96 = mul i64 %v95, 4
  br label %bb20
bb19:
  %v97 = mul i64 %v45, %v53
  %v98 = add i64 %v97, %v68
  %v99 = udiv i64 %v98, 8
  %v100 = urem i64 %v98, 8
  %v101 = mul i64 %v100, 4
  br label %bb20
bb20:
  %v102 = phi i64 [ %v94, %bb18 ], [ %v99, %bb19 ]
  %v103 = phi i64 [ %v96, %bb18 ], [ %v101, %bb19 ]
  %v104 = extractvalue { ptr, i64 } %v28, 1
  %v105 = icmp ult i64 %v102, %v104
  br i1 %v105, label %bb21, label %bb58
bb21:
  %v106 = extractvalue { ptr, i64 } %v28, 0
  %v107 = getelementptr inbounds i32, ptr %v106, i64 %v102
  %v108 = load i32, ptr %v107, align 4
  %v109 = trunc i64 %v103 to i32
  %v110 = and i32 %v109, 31
  %v111 = lshr i32 %v108, %v110
  %v112 = and i32 %v111, 15
  %v113 = trunc i32 %v112 to i8
  %v114 = insertvalue { i64, i64 } undef, i64 0, 0
  %v115 = insertvalue { i64, i64 } %v114, i64 %v50, 1
  %v116 = extractvalue { i64, i64 } %v115, 0
  %v117 = extractvalue { i64, i64 } %v115, 1
  %v118 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v116, i64 %v117, i64 8) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v118, ptr %v35, align 8
  br label %bb40
bb22:
  %v119 = phi float [ %v153, %bb32 ], [ %v67, %bb40 ]
  %v120 = phi i64 [ %v205, %bb32 ], [ %v191, %bb40 ]
  %v121 = phi i64 [ %v206, %bb32 ], [ %v194, %bb40 ]
  %v122 = add i64 %v196, 1
  %v123 = icmp eq i64 %v122, 0
  %v124 = select i1 %v123, i8 0, i8 1
  %v125 = insertvalue { i8, { { i64 } } } undef, i8 %v124, 0
  %v126 = insertvalue { i8, { { i64 } } } %v125, i64 %v122, 1, 0, 0
  %v127 = extractvalue { i8, { { i64 } } } %v126, 0
  %v128 = zext i8 %v127 to i64
  %v129 = icmp eq i64 %v128, 1
  %v130 = extractvalue { i8, { { i64 } } } %v126, 1
  %v131 = alloca { { i64 } }, align 8
  store { { i64 } } %v130, ptr %v131, align 8
  %v132 = load i64, ptr %v131, align 8
  %v133 = icmp ugt i64 %v121, 0
  %v134 = xor i1 %v133, 1
  br i1 %v134, label %bb42, label %bb41
bb23:
  unreachable
bb24:
  %v135 = extractvalue { i64, i64 } %v204, 1
  %v136 = add i64 %v71, %v135
  %v137 = xor i1 %v72, 1
  br i1 %v137, label %bb27, label %bb26
bb25:
  %v138 = add i64 %v68, 32
  br label %bb11
bb26:
  %v139 = zext i32 3 to i64
  %v140 = and i64 %v139, 63
  %v141 = lshr i64 %v136, %v140
  %v142 = mul i64 %v141, %v48
  %v143 = add i64 %v142, %v45
  br label %bb28
bb27:
  %v144 = mul i64 %v45, %v49
  %v145 = add i64 %v144, %v136
  %v146 = udiv i64 %v145, 8
  br label %bb28
bb28:
  %v147 = phi i64 [ %v143, %bb26 ], [ %v146, %bb27 ]
  %v148 = extractvalue { ptr, i64 } %v26, 1
  %v149 = icmp ult i64 %v147, %v148
  br i1 %v149, label %bb29, label %bb59
bb29:
  %v150 = extractvalue { ptr, i64 } %v26, 0
  %v151 = getelementptr inbounds i32, ptr %v150, i64 %v147
  %v152 = load i32, ptr %v151, align 4
  br label %bb30
bb30:
  %v153 = phi float [ %v119, %bb29 ], [ %v180, %bb33 ]
  %v154 = phi i64 [ 0, %bb29 ], [ %v216, %bb33 ]
  %v155 = icmp ult i64 %v154, 8
  %v156 = xor i1 %v155, 1
  br i1 %v156, label %bb46, label %bb45
bb31:
  %v157 = extractvalue { i64, i64 } %v215, 1
  %v158 = mul i64 %v157, 4
  %v159 = trunc i64 %v158 to i32
  %v160 = and i32 %v159, 31
  %v161 = lshr i32 %v152, %v160
  %v162 = and i32 %v161, 15
  %v163 = trunc i32 %v162 to i8
  %v164 = sitofp i8 %v163 to float
  %v165 = sitofp i8 %v113 to float
  %v166 = fadd contract float %v165, 1.0
  %v167 = fsub contract float %v164, %v166
  %v168 = fmul contract float %v167, %v89
  %v169 = add i64 %v136, %v157
  %v170 = extractvalue { ptr, i64 } %v29, 1
  %v171 = icmp ult i64 %v169, %v170
  br i1 %v171, label %bb33, label %bb60
bb32:
  br label %bb22
bb33:
  %v172 = extractvalue { ptr, i64 } %v29, 0
  %v173 = getelementptr inbounds i16, ptr %v172, i64 %v169
  %v174 = load i16, ptr %v173, align 2
  %v175 = zext i16 %v174 to i32
  %v176 = and i32 16, 31
  %v177 = shl i32 %v175, %v176
  %v178 = bitcast i32 %v177 to float
  %v179 = fmul contract float %v168, %v178
  %v180 = fadd contract float %v153, %v179
  br label %bb30
bb34:
  %v181 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v67, i32 16, i32 31) #0
  br label %bb49
bb35:
  %v182 = mul i64 %v47, %v48
  %v183 = add i64 %v182, %v45
  %v184 = extractvalue { ptr, i64 } %v25, 1
  %v185 = icmp ult i64 %v183, %v184
  br i1 %v185, label %bb36, label %bb61
bb36:
  %v186 = extractvalue { ptr, i64 } %v25, 0
  %v187 = getelementptr inbounds float, ptr %v186, i64 %v183
  store float %v229, ptr %v187, align 4
  br label %bb37
bb37:
  br label %bb38
bb38:
  ret void
bb39:
  %v188 = xor i1 %v72, 1
  br i1 %v188, label %bb19, label %bb18
bb40:
  %v189 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v190 = getelementptr inbounds { i64, i64 }, ptr %v189, i32 0, i32 0
  %v191 = load i64, ptr %v190, align 8
  %v192 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v193 = getelementptr inbounds { i64, i64 }, ptr %v192, i32 0, i32 1
  %v194 = load i64, ptr %v193, align 8
  %v195 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 1
  %v196 = load i64, ptr %v195, align 8
  %v197 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 2
  %v198 = load i1, ptr %v197, align 1
  br label %bb22
bb41:
  %v199 = add i64 %v120, %v132
  %v200 = sub i64 %v121, 1
  %v201 = insertvalue { i64, i64 } undef, i64 1, 0
  %v202 = insertvalue { i64, i64 } %v201, i64 %v120, 1
  br label %bb43
bb42:
  %v203 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb43
bb43:
  %v204 = phi { i64, i64 } [ %v202, %bb41 ], [ %v203, %bb42 ]
  %v205 = phi i64 [ %v199, %bb41 ], [ %v120, %bb42 ]
  %v206 = phi i64 [ %v200, %bb41 ], [ %v121, %bb42 ]
  %v207 = extractvalue { i64, i64 } %v204, 0
  %v208 = bitcast i64 %v207 to i64
  %v209 = icmp eq i64 %v208, 0
  br i1 %v209, label %bb25, label %bb44
bb44:
  %v210 = icmp eq i64 %v208, 1
  br i1 %v210, label %bb24, label %bb23
bb45:
  %v211 = add i64 %v154, 1
  %v212 = insertvalue { i64, i64 } undef, i64 1, 0
  %v213 = insertvalue { i64, i64 } %v212, i64 %v154, 1
  br label %bb47
bb46:
  %v214 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb47
bb47:
  %v215 = phi { i64, i64 } [ %v213, %bb45 ], [ %v214, %bb46 ]
  %v216 = phi i64 [ %v211, %bb45 ], [ %v154, %bb46 ]
  %v217 = extractvalue { i64, i64 } %v215, 0
  %v218 = bitcast i64 %v217 to i64
  %v219 = icmp eq i64 %v218, 0
  br i1 %v219, label %bb32, label %bb48
bb48:
  %v220 = icmp eq i64 %v218, 1
  br i1 %v220, label %bb31, label %bb23
bb49:
  %v221 = fadd contract float %v67, %v181
  %v222 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v221, i32 8, i32 31) #0
  br label %bb50
bb50:
  %v223 = fadd contract float %v221, %v222
  %v224 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v223, i32 4, i32 31) #0
  br label %bb51
bb51:
  %v225 = fadd contract float %v223, %v224
  %v226 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v225, i32 2, i32 31) #0
  br label %bb52
bb52:
  %v227 = fadd contract float %v225, %v226
  %v228 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v227, i32 1, i32 31) #0
  br label %bb53
bb53:
  %v229 = fadd contract float %v227, %v228
  %v230 = icmp eq i64 %v38, 0
  br i1 %v230, label %bb35, label %bb37
bb54:
  unreachable
bb55:
  unreachable
bb56:
  unreachable
bb57:
  unreachable
bb58:
  unreachable
bb59:
  unreachable
bb60:
  unreachable
bb61:
  unreachable
}

declare void @llvm.nvvm.barrier0() #0

define void @int4_gemm_v3_ksplit_sm(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v7, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v9, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v16, %entry ]
  %v26 = phi { ptr, i64 } [ %v18, %entry ]
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v36 = alloca [4 x i32], align 4
  %v37 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb2
bb2:
  %v40 = zext i32 %v39 to i64
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb3
bb3:
  %v42 = mul i32 %v41, 64
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb4
bb4:
  %v44 = add i32 %v42, %v43
  %v45 = zext i32 %v44 to i64
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb5
bb5:
  %v47 = zext i32 %v46 to i64
  %v48 = zext i32 %v30 to i64
  %v49 = zext i32 %v31 to i64
  %v50 = zext i32 %v32 to i64
  %v51 = zext i32 %v34 to i64
  %v52 = icmp eq i64 %v50, 0
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb6, label %bb74
bb6:
  %v54 = udiv i64 %v49, %v50
  %v55 = add i64 %v54, %v51
  %v56 = sub i64 %v55, 1
  %v57 = icmp eq i64 %v51, 0
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb7, label %bb75
bb7:
  %v59 = udiv i64 %v56, %v51
  %v60 = mul i64 %v47, %v59
  %v61 = icmp uge i64 %v60, %v54
  %v62 = xor i1 %v61, 1
  br i1 %v62, label %bb12, label %bb8
bb8:
  %v63 = icmp ult i64 %v45, %v48
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb11, label %bb9
bb9:
  %v65 = mul i64 %v47, %v48
  %v66 = add i64 %v65, %v45
  %v67 = extractvalue { ptr, i64 } %v25, 1
  %v68 = icmp ult i64 %v66, %v67
  br i1 %v68, label %bb10, label %bb76
bb10:
  %v69 = extractvalue { ptr, i64 } %v25, 0
  %v70 = getelementptr inbounds float, ptr %v69, i64 %v66
  store float 0.0, ptr %v70, align 4
  br label %bb11
bb11:
  br label %bb54
bb12:
  %v71 = add i64 %v60, %v59
  %v72 = icmp ugt i64 %v71, %v54
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb14, label %bb13
bb13:
  br label %bb15
bb14:
  br label %bb15
bb15:
  %v74 = phi i64 [ %v54, %bb13 ], [ %v71, %bb14 ]
  br label %bb16
bb16:
  %v75 = add i64 %v48, 7
  %v76 = udiv i64 %v75, 8
  %v77 = udiv i64 %v50, 8
  br label %bb17
bb17:
  %v78 = phi float [ 0.0, %bb16 ], [ %v694, %bb50 ]
  %v79 = phi float [ 0.0, %bb16 ], [ %v695, %bb50 ]
  %v80 = phi float [ 0.0, %bb16 ], [ %v696, %bb50 ]
  %v81 = phi float [ 0.0, %bb16 ], [ %v697, %bb50 ]
  %v82 = phi i64 [ %v60, %bb16 ], [ %v710, %bb50 ]
  %v83 = icmp ult i64 %v82, %v74
  %v84 = xor i1 %v83, 1
  br i1 %v84, label %bb56, label %bb55
bb18:
  unreachable
bb19:
  %v85 = extractvalue { i64, i64 } %v709, 1
  %v86 = mul i64 %v85, %v50
  br label %bb21
bb20:
  %v87 = fadd contract float %v78, %v79
  %v88 = fadd contract float %v87, %v80
  %v89 = fadd contract float %v88, %v81
  %v90 = icmp ult i64 %v45, %v48
  %v91 = xor i1 %v90, 1
  br i1 %v91, label %bb53, label %bb51
bb21:
  %v92 = phi i64 [ %v40, %bb19 ], [ %v103, %bb22 ]
  %v93 = icmp ult i64 %v92, %v50
  %v94 = xor i1 %v93, 1
  br i1 %v94, label %bb23, label %bb22
bb22:
  %v95 = add i64 %v86, %v92
  %v96 = extractvalue { ptr, i64 } %v29, 1
  %v97 = icmp ult i64 %v95, %v96
  %v98 = extractvalue { ptr, i64 } %v29, 0
  %v99 = getelementptr inbounds i16, ptr %v98, i64 %v95
  %v100 = load i16, ptr %v99, align 2
  %v101 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v92
  %v102 = addrspacecast ptr addrspace(3) %v101 to ptr
  store i16 %v100, ptr %v102, align 2
  %v103 = add i64 %v92, 64
  br label %bb21
bb23:
  call void @llvm.nvvm.barrier0() #0
  br label %bb24
bb24:
  %v105 = icmp ne i32 %v33, 0
  %v106 = icmp eq i32 %v33, 0
  br i1 %v106, label %bb26, label %bb25
bb25:
  %v107 = mul i64 %v85, %v48
  %v108 = add i64 %v107, %v45
  %v109 = extractvalue { ptr, i64 } %v27, 1
  %v110 = icmp ult i64 %v108, %v109
  %v111 = extractvalue { ptr, i64 } %v27, 0
  %v112 = getelementptr inbounds i16, ptr %v111, i64 %v108
  %v113 = load i16, ptr %v112, align 2
  br label %bb27
bb26:
  %v114 = mul i64 %v45, %v54
  %v115 = add i64 %v114, %v85
  %v116 = extractvalue { ptr, i64 } %v27, 1
  %v117 = icmp ult i64 %v115, %v116
  %v118 = extractvalue { ptr, i64 } %v27, 0
  %v119 = getelementptr inbounds i16, ptr %v118, i64 %v115
  %v120 = load i16, ptr %v119, align 2
  br label %bb27
bb27:
  %v121 = phi i16 [ %v113, %bb25 ], [ %v120, %bb26 ]
  %v122 = call float @f16_to_f32(i16 %v121) #0
  br label %bb59
bb28:
  %v123 = mul i64 %v85, %v76
  %v124 = udiv i64 %v45, 8
  %v125 = add i64 %v123, %v124
  %v126 = urem i64 %v45, 8
  %v127 = mul i64 %v126, 4
  br label %bb30
bb29:
  %v128 = mul i64 %v45, %v54
  %v129 = add i64 %v128, %v85
  %v130 = udiv i64 %v129, 8
  %v131 = urem i64 %v129, 8
  %v132 = mul i64 %v131, 4
  br label %bb30
bb30:
  %v133 = phi i64 [ %v125, %bb28 ], [ %v130, %bb29 ]
  %v134 = phi i64 [ %v127, %bb28 ], [ %v132, %bb29 ]
  %v135 = extractvalue { ptr, i64 } %v28, 1
  %v136 = icmp ult i64 %v133, %v135
  %v137 = extractvalue { ptr, i64 } %v28, 0
  %v138 = getelementptr inbounds i32, ptr %v137, i64 %v133
  %v139 = load i32, ptr %v138, align 4
  %v140 = trunc i64 %v134 to i32
  %v141 = and i32 %v140, 31
  %v142 = lshr i32 %v139, %v141
  %v143 = and i32 %v142, 15
  %v144 = trunc i32 %v143 to i8
  %v145 = sitofp i8 %v144 to float
  %v146 = fadd contract float %v145, 1.0
  %v147 = fmul contract float %v146, %v122
  %v148 = icmp eq i32 %v33, 0
  br i1 %v148, label %bb31, label %bb42
bb31:
  %v149 = insertvalue { i64, i64 } undef, i64 0, 0
  %v150 = insertvalue { i64, i64 } %v149, i64 %v77, 1
  %v151 = extractvalue { i64, i64 } %v150, 0
  %v152 = extractvalue { i64, i64 } %v150, 1
  %v153 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v151, i64 %v152, i64 4) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v153, ptr %v35, align 8
  br label %bb60
bb32:
  %v154 = phi float [ %v183, %bb37 ], [ %v78, %bb60 ]
  %v155 = phi float [ %v184, %bb37 ], [ %v79, %bb60 ]
  %v156 = phi float [ %v185, %bb37 ], [ %v80, %bb60 ]
  %v157 = phi float [ %v186, %bb37 ], [ %v81, %bb60 ]
  %v158 = phi i64 [ %v732, %bb37 ], [ %v718, %bb60 ]
  %v159 = phi i64 [ %v733, %bb37 ], [ %v721, %bb60 ]
  %v160 = add i64 %v723, 1
  %v161 = icmp eq i64 %v160, 0
  %v162 = select i1 %v161, i8 0, i8 1
  %v163 = insertvalue { i8, { { i64 } } } undef, i8 %v162, 0
  %v164 = insertvalue { i8, { { i64 } } } %v163, i64 %v160, 1, 0, 0
  %v165 = extractvalue { i8, { { i64 } } } %v164, 0
  %v166 = zext i8 %v165 to i64
  %v167 = icmp eq i64 %v166, 1
  %v168 = extractvalue { i8, { { i64 } } } %v164, 1
  %v169 = alloca { { i64 } }, align 8
  store { { i64 } } %v168, ptr %v169, align 8
  %v170 = load i64, ptr %v169, align 8
  %v171 = icmp ugt i64 %v159, 0
  %v172 = xor i1 %v171, 1
  br i1 %v172, label %bb62, label %bb61
bb33:
  %v173 = extractvalue { i64, i64 } %v731, 1
  %v174 = mul i64 %v173, 8
  %v175 = add i64 %v86, %v174
  %v176 = mul i64 %v45, %v49
  %v177 = add i64 %v176, %v175
  %v178 = udiv i64 %v177, 8
  %v179 = extractvalue { ptr, i64 } %v26, 0
  %v180 = getelementptr inbounds i32, ptr %v179, i64 %v178
  %v181 = bitcast ptr %v180 to ptr
  %v182 = load [4 x i32], ptr %v181, align 4
  store [4 x i32] %v182, ptr %v36, align 4
  br label %bb35
bb34:
  br label %bb49
bb35:
  %v183 = phi float [ %v154, %bb33 ], [ %v371, %bb41 ]
  %v184 = phi float [ %v155, %bb33 ], [ %v372, %bb41 ]
  %v185 = phi float [ %v156, %bb33 ], [ %v373, %bb41 ]
  %v186 = phi float [ %v157, %bb33 ], [ %v374, %bb41 ]
  %v187 = phi i64 [ 0, %bb33 ], [ %v743, %bb41 ]
  %v188 = icmp ult i64 %v187, 4
  %v189 = xor i1 %v188, 1
  br i1 %v189, label %bb66, label %bb65
bb36:
  %v190 = extractvalue { i64, i64 } %v742, 1
  %v191 = icmp ult i64 %v190, 4
  br i1 %v191, label %bb38, label %bb77
bb37:
  br label %bb32
bb38:
  %v192 = getelementptr inbounds [4 x i32], ptr %v36, i32 0, i64 %v190
  %v193 = load i32, ptr %v192, align 4
  %v194 = add i64 %v173, %v190
  %v195 = mul i64 %v194, 8
  %v196 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v195
  %v197 = addrspacecast ptr addrspace(3) %v196 to ptr
  %v198 = load i16, ptr %v197, align 2
  %v199 = zext i16 %v198 to i32
  %v200 = and i32 16, 31
  %v201 = shl i32 %v199, %v200
  %v202 = bitcast i32 %v201 to float
  %v203 = add i64 %v195, 1
  %v204 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v203
  %v205 = addrspacecast ptr addrspace(3) %v204 to ptr
  %v206 = load i16, ptr %v205, align 2
  %v207 = zext i16 %v206 to i32
  %v208 = and i32 16, 31
  %v209 = shl i32 %v207, %v208
  %v210 = bitcast i32 %v209 to float
  %v211 = add i64 %v195, 2
  %v212 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v211
  %v213 = addrspacecast ptr addrspace(3) %v212 to ptr
  %v214 = load i16, ptr %v213, align 2
  %v215 = zext i16 %v214 to i32
  %v216 = and i32 16, 31
  %v217 = shl i32 %v215, %v216
  %v218 = bitcast i32 %v217 to float
  %v219 = add i64 %v195, 3
  %v220 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v219
  %v221 = addrspacecast ptr addrspace(3) %v220 to ptr
  %v222 = load i16, ptr %v221, align 2
  %v223 = zext i16 %v222 to i32
  %v224 = and i32 16, 31
  %v225 = shl i32 %v223, %v224
  %v226 = bitcast i32 %v225 to float
  %v227 = add i64 %v195, 4
  %v228 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v227
  %v229 = addrspacecast ptr addrspace(3) %v228 to ptr
  %v230 = load i16, ptr %v229, align 2
  %v231 = zext i16 %v230 to i32
  %v232 = and i32 16, 31
  %v233 = shl i32 %v231, %v232
  %v234 = bitcast i32 %v233 to float
  %v235 = add i64 %v195, 5
  %v236 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v235
  %v237 = addrspacecast ptr addrspace(3) %v236 to ptr
  %v238 = load i16, ptr %v237, align 2
  %v239 = zext i16 %v238 to i32
  %v240 = and i32 16, 31
  %v241 = shl i32 %v239, %v240
  %v242 = bitcast i32 %v241 to float
  %v243 = add i64 %v195, 6
  %v244 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v243
  %v245 = addrspacecast ptr addrspace(3) %v244 to ptr
  %v246 = load i16, ptr %v245, align 2
  %v247 = zext i16 %v246 to i32
  %v248 = and i32 16, 31
  %v249 = shl i32 %v247, %v248
  %v250 = bitcast i32 %v249 to float
  %v251 = add i64 %v195, 7
  %v252 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v251
  %v253 = addrspacecast ptr addrspace(3) %v252 to ptr
  %v254 = load i16, ptr %v253, align 2
  %v255 = zext i16 %v254 to i32
  %v256 = and i32 16, 31
  %v257 = shl i32 %v255, %v256
  %v258 = bitcast i32 %v257 to float
  %v259 = and i32 %v193, 15
  %v260 = trunc i32 %v259 to i8
  %v261 = and i32 4, 31
  %v262 = lshr i32 %v193, %v261
  %v263 = and i32 %v262, 15
  %v264 = trunc i32 %v263 to i8
  %v265 = and i32 8, 31
  %v266 = lshr i32 %v193, %v265
  %v267 = and i32 %v266, 15
  %v268 = trunc i32 %v267 to i8
  %v269 = and i32 12, 31
  %v270 = lshr i32 %v193, %v269
  %v271 = and i32 %v270, 15
  %v272 = trunc i32 %v271 to i8
  %v273 = and i32 16, 31
  %v274 = lshr i32 %v193, %v273
  %v275 = and i32 %v274, 15
  %v276 = trunc i32 %v275 to i8
  %v277 = and i32 20, 31
  %v278 = lshr i32 %v193, %v277
  %v279 = and i32 %v278, 15
  %v280 = trunc i32 %v279 to i8
  %v281 = and i32 24, 31
  %v282 = lshr i32 %v193, %v281
  %v283 = and i32 %v282, 15
  %v284 = trunc i32 %v283 to i8
  %v285 = and i32 28, 31
  %v286 = lshr i32 %v193, %v285
  %v287 = and i32 %v286, 15
  %v288 = trunc i32 %v287 to i8
  %v289 = urem i64 %v190, 2
  %v290 = icmp eq i64 %v289, 0
  br i1 %v290, label %bb39, label %bb40
bb39:
  %v291 = sitofp i8 %v260 to float
  %v292 = fmul contract float %v291, %v122
  %v293 = fsub contract float %v292, %v147
  %v294 = fmul contract float %v293, %v202
  %v295 = fadd contract float %v183, %v294
  %v296 = sitofp i8 %v264 to float
  %v297 = fmul contract float %v296, %v122
  %v298 = fsub contract float %v297, %v147
  %v299 = fmul contract float %v298, %v210
  %v300 = fadd contract float %v184, %v299
  %v301 = sitofp i8 %v268 to float
  %v302 = fmul contract float %v301, %v122
  %v303 = fsub contract float %v302, %v147
  %v304 = fmul contract float %v303, %v218
  %v305 = fadd contract float %v295, %v304
  %v306 = sitofp i8 %v272 to float
  %v307 = fmul contract float %v306, %v122
  %v308 = fsub contract float %v307, %v147
  %v309 = fmul contract float %v308, %v226
  %v310 = fadd contract float %v300, %v309
  %v311 = sitofp i8 %v276 to float
  %v312 = fmul contract float %v311, %v122
  %v313 = fsub contract float %v312, %v147
  %v314 = fmul contract float %v313, %v234
  %v315 = fadd contract float %v305, %v314
  %v316 = sitofp i8 %v280 to float
  %v317 = fmul contract float %v316, %v122
  %v318 = fsub contract float %v317, %v147
  %v319 = fmul contract float %v318, %v242
  %v320 = fadd contract float %v310, %v319
  %v321 = sitofp i8 %v284 to float
  %v322 = fmul contract float %v321, %v122
  %v323 = fsub contract float %v322, %v147
  %v324 = fmul contract float %v323, %v250
  %v325 = fadd contract float %v315, %v324
  %v326 = sitofp i8 %v288 to float
  %v327 = fmul contract float %v326, %v122
  %v328 = fsub contract float %v327, %v147
  %v329 = fmul contract float %v328, %v258
  %v330 = fadd contract float %v320, %v329
  br label %bb41
bb40:
  %v331 = sitofp i8 %v260 to float
  %v332 = fmul contract float %v331, %v122
  %v333 = fsub contract float %v332, %v147
  %v334 = fmul contract float %v333, %v202
  %v335 = fadd contract float %v185, %v334
  %v336 = sitofp i8 %v264 to float
  %v337 = fmul contract float %v336, %v122
  %v338 = fsub contract float %v337, %v147
  %v339 = fmul contract float %v338, %v210
  %v340 = fadd contract float %v186, %v339
  %v341 = sitofp i8 %v268 to float
  %v342 = fmul contract float %v341, %v122
  %v343 = fsub contract float %v342, %v147
  %v344 = fmul contract float %v343, %v218
  %v345 = fadd contract float %v335, %v344
  %v346 = sitofp i8 %v272 to float
  %v347 = fmul contract float %v346, %v122
  %v348 = fsub contract float %v347, %v147
  %v349 = fmul contract float %v348, %v226
  %v350 = fadd contract float %v340, %v349
  %v351 = sitofp i8 %v276 to float
  %v352 = fmul contract float %v351, %v122
  %v353 = fsub contract float %v352, %v147
  %v354 = fmul contract float %v353, %v234
  %v355 = fadd contract float %v345, %v354
  %v356 = sitofp i8 %v280 to float
  %v357 = fmul contract float %v356, %v122
  %v358 = fsub contract float %v357, %v147
  %v359 = fmul contract float %v358, %v242
  %v360 = fadd contract float %v350, %v359
  %v361 = sitofp i8 %v284 to float
  %v362 = fmul contract float %v361, %v122
  %v363 = fsub contract float %v362, %v147
  %v364 = fmul contract float %v363, %v250
  %v365 = fadd contract float %v355, %v364
  %v366 = sitofp i8 %v288 to float
  %v367 = fmul contract float %v366, %v122
  %v368 = fsub contract float %v367, %v147
  %v369 = fmul contract float %v368, %v258
  %v370 = fadd contract float %v360, %v369
  br label %bb41
bb41:
  %v371 = phi float [ %v325, %bb39 ], [ %v183, %bb40 ]
  %v372 = phi float [ %v330, %bb39 ], [ %v184, %bb40 ]
  %v373 = phi float [ %v185, %bb39 ], [ %v365, %bb40 ]
  %v374 = phi float [ %v186, %bb39 ], [ %v370, %bb40 ]
  br label %bb35
bb42:
  %v375 = insertvalue { i64, i64 } undef, i64 0, 0
  %v376 = insertvalue { i64, i64 } %v375, i64 %v77, 1
  %v377 = extractvalue { i64, i64 } %v376, 0
  %v378 = extractvalue { i64, i64 } %v376, 1
  %v379 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v377, i64 %v378, i64 2) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v379, ptr %v37, align 8
  br label %bb69
bb43:
  %v380 = phi float [ %v539, %bb48 ], [ %v78, %bb69 ]
  %v381 = phi float [ %v544, %bb48 ], [ %v79, %bb69 ]
  %v382 = phi float [ %v692, %bb48 ], [ %v80, %bb69 ]
  %v383 = phi float [ %v693, %bb48 ], [ %v81, %bb69 ]
  %v384 = phi i64 [ %v764, %bb48 ], [ %v750, %bb69 ]
  %v385 = phi i64 [ %v765, %bb48 ], [ %v753, %bb69 ]
  %v386 = add i64 %v755, 1
  %v387 = icmp eq i64 %v386, 0
  %v388 = select i1 %v387, i8 0, i8 1
  %v389 = insertvalue { i8, { { i64 } } } undef, i8 %v388, 0
  %v390 = insertvalue { i8, { { i64 } } } %v389, i64 %v386, 1, 0, 0
  %v391 = extractvalue { i8, { { i64 } } } %v390, 0
  %v392 = zext i8 %v391 to i64
  %v393 = icmp eq i64 %v392, 1
  %v394 = extractvalue { i8, { { i64 } } } %v390, 1
  %v395 = alloca { { i64 } }, align 8
  store { { i64 } } %v394, ptr %v395, align 8
  %v396 = load i64, ptr %v395, align 8
  %v397 = icmp ugt i64 %v385, 0
  %v398 = xor i1 %v397, 1
  br i1 %v398, label %bb71, label %bb70
bb44:
  %v399 = extractvalue { i64, i64 } %v763, 1
  %v400 = mul i64 %v399, 8
  %v401 = add i64 %v86, %v400
  %v402 = zext i32 3 to i64
  %v403 = and i64 %v402, 63
  %v404 = lshr i64 %v401, %v403
  %v405 = mul i64 %v404, %v48
  %v406 = add i64 %v405, %v45
  %v407 = extractvalue { ptr, i64 } %v26, 1
  %v408 = icmp ult i64 %v406, %v407
  %v409 = extractvalue { ptr, i64 } %v26, 0
  %v410 = getelementptr inbounds i32, ptr %v409, i64 %v406
  %v411 = load i32, ptr %v410, align 4
  %v412 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v400
  %v413 = addrspacecast ptr addrspace(3) %v412 to ptr
  %v414 = load i16, ptr %v413, align 2
  %v415 = zext i16 %v414 to i32
  %v416 = and i32 16, 31
  %v417 = shl i32 %v415, %v416
  %v418 = bitcast i32 %v417 to float
  %v419 = add i64 %v400, 1
  %v420 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v419
  %v421 = addrspacecast ptr addrspace(3) %v420 to ptr
  %v422 = load i16, ptr %v421, align 2
  %v423 = zext i16 %v422 to i32
  %v424 = and i32 16, 31
  %v425 = shl i32 %v423, %v424
  %v426 = bitcast i32 %v425 to float
  %v427 = add i64 %v400, 2
  %v428 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v427
  %v429 = addrspacecast ptr addrspace(3) %v428 to ptr
  %v430 = load i16, ptr %v429, align 2
  %v431 = zext i16 %v430 to i32
  %v432 = and i32 16, 31
  %v433 = shl i32 %v431, %v432
  %v434 = bitcast i32 %v433 to float
  %v435 = add i64 %v400, 3
  %v436 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v435
  %v437 = addrspacecast ptr addrspace(3) %v436 to ptr
  %v438 = load i16, ptr %v437, align 2
  %v439 = zext i16 %v438 to i32
  %v440 = and i32 16, 31
  %v441 = shl i32 %v439, %v440
  %v442 = bitcast i32 %v441 to float
  %v443 = add i64 %v400, 4
  %v444 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v443
  %v445 = addrspacecast ptr addrspace(3) %v444 to ptr
  %v446 = load i16, ptr %v445, align 2
  %v447 = zext i16 %v446 to i32
  %v448 = and i32 16, 31
  %v449 = shl i32 %v447, %v448
  %v450 = bitcast i32 %v449 to float
  %v451 = add i64 %v400, 5
  %v452 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v451
  %v453 = addrspacecast ptr addrspace(3) %v452 to ptr
  %v454 = load i16, ptr %v453, align 2
  %v455 = zext i16 %v454 to i32
  %v456 = and i32 16, 31
  %v457 = shl i32 %v455, %v456
  %v458 = bitcast i32 %v457 to float
  %v459 = add i64 %v400, 6
  %v460 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v459
  %v461 = addrspacecast ptr addrspace(3) %v460 to ptr
  %v462 = load i16, ptr %v461, align 2
  %v463 = zext i16 %v462 to i32
  %v464 = and i32 16, 31
  %v465 = shl i32 %v463, %v464
  %v466 = bitcast i32 %v465 to float
  %v467 = add i64 %v400, 7
  %v468 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v467
  %v469 = addrspacecast ptr addrspace(3) %v468 to ptr
  %v470 = load i16, ptr %v469, align 2
  %v471 = zext i16 %v470 to i32
  %v472 = and i32 16, 31
  %v473 = shl i32 %v471, %v472
  %v474 = bitcast i32 %v473 to float
  %v475 = and i32 %v411, 15
  %v476 = trunc i32 %v475 to i8
  %v477 = and i32 4, 31
  %v478 = lshr i32 %v411, %v477
  %v479 = and i32 %v478, 15
  %v480 = trunc i32 %v479 to i8
  %v481 = and i32 8, 31
  %v482 = lshr i32 %v411, %v481
  %v483 = and i32 %v482, 15
  %v484 = trunc i32 %v483 to i8
  %v485 = and i32 12, 31
  %v486 = lshr i32 %v411, %v485
  %v487 = and i32 %v486, 15
  %v488 = trunc i32 %v487 to i8
  %v489 = and i32 16, 31
  %v490 = lshr i32 %v411, %v489
  %v491 = and i32 %v490, 15
  %v492 = trunc i32 %v491 to i8
  %v493 = and i32 20, 31
  %v494 = lshr i32 %v411, %v493
  %v495 = and i32 %v494, 15
  %v496 = trunc i32 %v495 to i8
  %v497 = and i32 24, 31
  %v498 = lshr i32 %v411, %v497
  %v499 = and i32 %v498, 15
  %v500 = trunc i32 %v499 to i8
  %v501 = and i32 28, 31
  %v502 = lshr i32 %v411, %v501
  %v503 = and i32 %v502, 15
  %v504 = trunc i32 %v503 to i8
  %v505 = sitofp i8 %v476 to float
  %v506 = fmul contract float %v505, %v122
  %v507 = fsub contract float %v506, %v147
  %v508 = fmul contract float %v507, %v418
  %v509 = fadd contract float %v380, %v508
  %v510 = sitofp i8 %v480 to float
  %v511 = fmul contract float %v510, %v122
  %v512 = fsub contract float %v511, %v147
  %v513 = fmul contract float %v512, %v426
  %v514 = fadd contract float %v381, %v513
  %v515 = sitofp i8 %v484 to float
  %v516 = fmul contract float %v515, %v122
  %v517 = fsub contract float %v516, %v147
  %v518 = fmul contract float %v517, %v434
  %v519 = fadd contract float %v509, %v518
  %v520 = sitofp i8 %v488 to float
  %v521 = fmul contract float %v520, %v122
  %v522 = fsub contract float %v521, %v147
  %v523 = fmul contract float %v522, %v442
  %v524 = fadd contract float %v514, %v523
  %v525 = sitofp i8 %v492 to float
  %v526 = fmul contract float %v525, %v122
  %v527 = fsub contract float %v526, %v147
  %v528 = fmul contract float %v527, %v450
  %v529 = fadd contract float %v519, %v528
  %v530 = sitofp i8 %v496 to float
  %v531 = fmul contract float %v530, %v122
  %v532 = fsub contract float %v531, %v147
  %v533 = fmul contract float %v532, %v458
  %v534 = fadd contract float %v524, %v533
  %v535 = sitofp i8 %v500 to float
  %v536 = fmul contract float %v535, %v122
  %v537 = fsub contract float %v536, %v147
  %v538 = fmul contract float %v537, %v466
  %v539 = fadd contract float %v529, %v538
  %v540 = sitofp i8 %v504 to float
  %v541 = fmul contract float %v540, %v122
  %v542 = fsub contract float %v541, %v147
  %v543 = fmul contract float %v542, %v474
  %v544 = fadd contract float %v534, %v543
  %v545 = add i64 %v399, 1
  %v546 = icmp ult i64 %v545, %v77
  %v547 = xor i1 %v546, 1
  br i1 %v547, label %bb47, label %bb46
bb45:
  br label %bb49
bb46:
  %v548 = add i64 %v401, 8
  %v549 = zext i32 3 to i64
  %v550 = and i64 %v549, 63
  %v551 = lshr i64 %v548, %v550
  %v552 = mul i64 %v551, %v48
  %v553 = add i64 %v552, %v45
  %v554 = icmp ult i64 %v553, %v407
  %v555 = extractvalue { ptr, i64 } %v26, 0
  %v556 = getelementptr inbounds i32, ptr %v555, i64 %v553
  %v557 = load i32, ptr %v556, align 4
  %v558 = add i64 %v400, 8
  %v559 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v558
  %v560 = addrspacecast ptr addrspace(3) %v559 to ptr
  %v561 = load i16, ptr %v560, align 2
  %v562 = zext i16 %v561 to i32
  %v563 = and i32 16, 31
  %v564 = shl i32 %v562, %v563
  %v565 = bitcast i32 %v564 to float
  %v566 = add i64 %v400, 9
  %v567 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v566
  %v568 = addrspacecast ptr addrspace(3) %v567 to ptr
  %v569 = load i16, ptr %v568, align 2
  %v570 = zext i16 %v569 to i32
  %v571 = and i32 16, 31
  %v572 = shl i32 %v570, %v571
  %v573 = bitcast i32 %v572 to float
  %v574 = add i64 %v400, 10
  %v575 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v574
  %v576 = addrspacecast ptr addrspace(3) %v575 to ptr
  %v577 = load i16, ptr %v576, align 2
  %v578 = zext i16 %v577 to i32
  %v579 = and i32 16, 31
  %v580 = shl i32 %v578, %v579
  %v581 = bitcast i32 %v580 to float
  %v582 = add i64 %v400, 11
  %v583 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v582
  %v584 = addrspacecast ptr addrspace(3) %v583 to ptr
  %v585 = load i16, ptr %v584, align 2
  %v586 = zext i16 %v585 to i32
  %v587 = and i32 16, 31
  %v588 = shl i32 %v586, %v587
  %v589 = bitcast i32 %v588 to float
  %v590 = add i64 %v400, 12
  %v591 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v590
  %v592 = addrspacecast ptr addrspace(3) %v591 to ptr
  %v593 = load i16, ptr %v592, align 2
  %v594 = zext i16 %v593 to i32
  %v595 = and i32 16, 31
  %v596 = shl i32 %v594, %v595
  %v597 = bitcast i32 %v596 to float
  %v598 = add i64 %v400, 13
  %v599 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v598
  %v600 = addrspacecast ptr addrspace(3) %v599 to ptr
  %v601 = load i16, ptr %v600, align 2
  %v602 = zext i16 %v601 to i32
  %v603 = and i32 16, 31
  %v604 = shl i32 %v602, %v603
  %v605 = bitcast i32 %v604 to float
  %v606 = add i64 %v400, 14
  %v607 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v606
  %v608 = addrspacecast ptr addrspace(3) %v607 to ptr
  %v609 = load i16, ptr %v608, align 2
  %v610 = zext i16 %v609 to i32
  %v611 = and i32 16, 31
  %v612 = shl i32 %v610, %v611
  %v613 = bitcast i32 %v612 to float
  %v614 = add i64 %v400, 15
  %v615 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_v3_ksplit_sm, i64 %v614
  %v616 = addrspacecast ptr addrspace(3) %v615 to ptr
  %v617 = load i16, ptr %v616, align 2
  %v618 = zext i16 %v617 to i32
  %v619 = and i32 16, 31
  %v620 = shl i32 %v618, %v619
  %v621 = bitcast i32 %v620 to float
  %v622 = and i32 %v557, 15
  %v623 = trunc i32 %v622 to i8
  %v624 = and i32 4, 31
  %v625 = lshr i32 %v557, %v624
  %v626 = and i32 %v625, 15
  %v627 = trunc i32 %v626 to i8
  %v628 = and i32 8, 31
  %v629 = lshr i32 %v557, %v628
  %v630 = and i32 %v629, 15
  %v631 = trunc i32 %v630 to i8
  %v632 = and i32 12, 31
  %v633 = lshr i32 %v557, %v632
  %v634 = and i32 %v633, 15
  %v635 = trunc i32 %v634 to i8
  %v636 = and i32 16, 31
  %v637 = lshr i32 %v557, %v636
  %v638 = and i32 %v637, 15
  %v639 = trunc i32 %v638 to i8
  %v640 = and i32 20, 31
  %v641 = lshr i32 %v557, %v640
  %v642 = and i32 %v641, 15
  %v643 = trunc i32 %v642 to i8
  %v644 = and i32 24, 31
  %v645 = lshr i32 %v557, %v644
  %v646 = and i32 %v645, 15
  %v647 = trunc i32 %v646 to i8
  %v648 = and i32 28, 31
  %v649 = lshr i32 %v557, %v648
  %v650 = and i32 %v649, 15
  %v651 = trunc i32 %v650 to i8
  %v652 = sitofp i8 %v623 to float
  %v653 = fmul contract float %v652, %v122
  %v654 = fsub contract float %v653, %v147
  %v655 = fmul contract float %v654, %v565
  %v656 = fadd contract float %v382, %v655
  %v657 = sitofp i8 %v627 to float
  %v658 = fmul contract float %v657, %v122
  %v659 = fsub contract float %v658, %v147
  %v660 = fmul contract float %v659, %v573
  %v661 = fadd contract float %v383, %v660
  %v662 = sitofp i8 %v631 to float
  %v663 = fmul contract float %v662, %v122
  %v664 = fsub contract float %v663, %v147
  %v665 = fmul contract float %v664, %v581
  %v666 = fadd contract float %v656, %v665
  %v667 = sitofp i8 %v635 to float
  %v668 = fmul contract float %v667, %v122
  %v669 = fsub contract float %v668, %v147
  %v670 = fmul contract float %v669, %v589
  %v671 = fadd contract float %v661, %v670
  %v672 = sitofp i8 %v639 to float
  %v673 = fmul contract float %v672, %v122
  %v674 = fsub contract float %v673, %v147
  %v675 = fmul contract float %v674, %v597
  %v676 = fadd contract float %v666, %v675
  %v677 = sitofp i8 %v643 to float
  %v678 = fmul contract float %v677, %v122
  %v679 = fsub contract float %v678, %v147
  %v680 = fmul contract float %v679, %v605
  %v681 = fadd contract float %v671, %v680
  %v682 = sitofp i8 %v647 to float
  %v683 = fmul contract float %v682, %v122
  %v684 = fsub contract float %v683, %v147
  %v685 = fmul contract float %v684, %v613
  %v686 = fadd contract float %v676, %v685
  %v687 = sitofp i8 %v651 to float
  %v688 = fmul contract float %v687, %v122
  %v689 = fsub contract float %v688, %v147
  %v690 = fmul contract float %v689, %v621
  %v691 = fadd contract float %v681, %v690
  br label %bb48
bb47:
  br label %bb48
bb48:
  %v692 = phi float [ %v686, %bb46 ], [ %v382, %bb47 ]
  %v693 = phi float [ %v691, %bb46 ], [ %v383, %bb47 ]
  br label %bb43
bb49:
  %v694 = phi float [ %v154, %bb34 ], [ %v380, %bb45 ]
  %v695 = phi float [ %v155, %bb34 ], [ %v381, %bb45 ]
  %v696 = phi float [ %v156, %bb34 ], [ %v382, %bb45 ]
  %v697 = phi float [ %v157, %bb34 ], [ %v383, %bb45 ]
  call void @llvm.nvvm.barrier0() #0
  br label %bb50
bb50:
  br label %bb17
bb51:
  %v699 = mul i64 %v47, %v48
  %v700 = add i64 %v699, %v45
  %v701 = extractvalue { ptr, i64 } %v25, 1
  %v702 = icmp ult i64 %v700, %v701
  br i1 %v702, label %bb52, label %bb78
bb52:
  %v703 = extractvalue { ptr, i64 } %v25, 0
  %v704 = getelementptr inbounds float, ptr %v703, i64 %v700
  store float %v89, ptr %v704, align 4
  br label %bb53
bb53:
  br label %bb54
bb54:
  ret void
bb55:
  %v705 = add i64 %v82, 1
  %v706 = insertvalue { i64, i64 } undef, i64 1, 0
  %v707 = insertvalue { i64, i64 } %v706, i64 %v82, 1
  br label %bb57
bb56:
  %v708 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb57
bb57:
  %v709 = phi { i64, i64 } [ %v707, %bb55 ], [ %v708, %bb56 ]
  %v710 = phi i64 [ %v705, %bb55 ], [ %v82, %bb56 ]
  %v711 = extractvalue { i64, i64 } %v709, 0
  %v712 = bitcast i64 %v711 to i64
  %v713 = icmp eq i64 %v712, 0
  br i1 %v713, label %bb20, label %bb58
bb58:
  %v714 = icmp eq i64 %v712, 1
  br i1 %v714, label %bb19, label %bb18
bb59:
  %v715 = xor i1 %v105, 1
  br i1 %v715, label %bb29, label %bb28
bb60:
  %v716 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v717 = getelementptr inbounds { i64, i64 }, ptr %v716, i32 0, i32 0
  %v718 = load i64, ptr %v717, align 8
  %v719 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v720 = getelementptr inbounds { i64, i64 }, ptr %v719, i32 0, i32 1
  %v721 = load i64, ptr %v720, align 8
  %v722 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 1
  %v723 = load i64, ptr %v722, align 8
  %v724 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 2
  %v725 = load i1, ptr %v724, align 1
  br label %bb32
bb61:
  %v726 = add i64 %v158, %v170
  %v727 = sub i64 %v159, 1
  %v728 = insertvalue { i64, i64 } undef, i64 1, 0
  %v729 = insertvalue { i64, i64 } %v728, i64 %v158, 1
  br label %bb63
bb62:
  %v730 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb63
bb63:
  %v731 = phi { i64, i64 } [ %v729, %bb61 ], [ %v730, %bb62 ]
  %v732 = phi i64 [ %v726, %bb61 ], [ %v158, %bb62 ]
  %v733 = phi i64 [ %v727, %bb61 ], [ %v159, %bb62 ]
  %v734 = extractvalue { i64, i64 } %v731, 0
  %v735 = bitcast i64 %v734 to i64
  %v736 = icmp eq i64 %v735, 0
  br i1 %v736, label %bb34, label %bb64
bb64:
  %v737 = icmp eq i64 %v735, 1
  br i1 %v737, label %bb33, label %bb18
bb65:
  %v738 = add i64 %v187, 1
  %v739 = insertvalue { i64, i64 } undef, i64 1, 0
  %v740 = insertvalue { i64, i64 } %v739, i64 %v187, 1
  br label %bb67
bb66:
  %v741 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb67
bb67:
  %v742 = phi { i64, i64 } [ %v740, %bb65 ], [ %v741, %bb66 ]
  %v743 = phi i64 [ %v738, %bb65 ], [ %v187, %bb66 ]
  %v744 = extractvalue { i64, i64 } %v742, 0
  %v745 = bitcast i64 %v744 to i64
  %v746 = icmp eq i64 %v745, 0
  br i1 %v746, label %bb37, label %bb68
bb68:
  %v747 = icmp eq i64 %v745, 1
  br i1 %v747, label %bb36, label %bb18
bb69:
  %v748 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 0
  %v749 = getelementptr inbounds { i64, i64 }, ptr %v748, i32 0, i32 0
  %v750 = load i64, ptr %v749, align 8
  %v751 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 0
  %v752 = getelementptr inbounds { i64, i64 }, ptr %v751, i32 0, i32 1
  %v753 = load i64, ptr %v752, align 8
  %v754 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 1
  %v755 = load i64, ptr %v754, align 8
  %v756 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 2
  %v757 = load i1, ptr %v756, align 1
  br label %bb43
bb70:
  %v758 = add i64 %v384, %v396
  %v759 = sub i64 %v385, 1
  %v760 = insertvalue { i64, i64 } undef, i64 1, 0
  %v761 = insertvalue { i64, i64 } %v760, i64 %v384, 1
  br label %bb72
bb71:
  %v762 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb72
bb72:
  %v763 = phi { i64, i64 } [ %v761, %bb70 ], [ %v762, %bb71 ]
  %v764 = phi i64 [ %v758, %bb70 ], [ %v384, %bb71 ]
  %v765 = phi i64 [ %v759, %bb70 ], [ %v385, %bb71 ]
  %v766 = extractvalue { i64, i64 } %v763, 0
  %v767 = bitcast i64 %v766 to i64
  %v768 = icmp eq i64 %v767, 0
  br i1 %v768, label %bb45, label %bb73
bb73:
  %v769 = icmp eq i64 %v767, 1
  br i1 %v769, label %bb44, label %bb18
bb74:
  unreachable
bb75:
  unreachable
bb76:
  unreachable
bb77:
  unreachable
bb78:
  unreachable
}

define void @int4_gemm_auto_round_tiled(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v7, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v9, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v16, %entry ]
  %v26 = phi { ptr, i64 } [ %v18, %entry ]
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v36 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v37 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb2
bb2:
  %v40 = zext i32 %v39 to i64
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb3
bb3:
  %v42 = mul i32 %v41, 64
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb4
bb4:
  %v44 = add i32 %v42, %v43
  %v45 = zext i32 %v44 to i64
  %v46 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb5
bb5:
  %v47 = zext i32 %v46 to i64
  %v48 = zext i32 %v31 to i64
  %v49 = zext i32 %v32 to i64
  %v50 = zext i32 %v33 to i64
  %v51 = zext i32 %v30 to i64
  %v52 = icmp uge i64 %v47, %v51
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb7, label %bb6
bb6:
  br label %bb44
bb7:
  br label %bb8
bb8:
  %v54 = insertvalue { i64, i64 } undef, i64 0, 0
  %v55 = insertvalue { i64, i64 } %v54, i64 %v49, 1
  %v56 = extractvalue { i64, i64 } %v55, 0
  %v57 = extractvalue { i64, i64 } %v55, 1
  %v58 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v56, i64 %v57, i64 %v50) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v58, ptr %v35, align 8
  br label %bb45
bb9:
  %v59 = phi float [ %v227, %bb41 ], [ 0.0, %bb45 ]
  %v60 = phi i64 [ %v253, %bb41 ], [ %v239, %bb45 ]
  %v61 = phi i64 [ %v254, %bb41 ], [ %v242, %bb45 ]
  %v62 = add i64 %v244, 1
  %v63 = icmp eq i64 %v62, 0
  %v64 = select i1 %v63, i8 0, i8 1
  %v65 = insertvalue { i8, { { i64 } } } undef, i8 %v64, 0
  %v66 = insertvalue { i8, { { i64 } } } %v65, i64 %v62, 1, 0, 0
  %v67 = extractvalue { i8, { { i64 } } } %v66, 0
  %v68 = zext i8 %v67 to i64
  %v69 = icmp eq i64 %v68, 1
  %v70 = extractvalue { i8, { { i64 } } } %v66, 1
  %v71 = alloca { { i64 } }, align 8
  store { { i64 } } %v70, ptr %v71, align 8
  %v72 = load i64, ptr %v71, align 8
  %v73 = icmp ugt i64 %v61, 0
  %v74 = xor i1 %v73, 1
  br i1 %v74, label %bb47, label %bb46
bb10:
  unreachable
bb11:
  %v75 = extractvalue { i64, i64 } %v252, 1
  %v76 = add i64 %v75, %v50
  %v77 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v76, i64 %v49) #0
  br label %bb13
bb12:
  %v78 = icmp ult i64 %v45, %v48
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb43, label %bb42
bb13:
  %v80 = sub i64 %v77, %v75
  %v81 = insertvalue { i64, i64 } undef, i64 %v40, 0
  %v82 = insertvalue { i64, i64 } %v81, i64 %v80, 1
  %v83 = extractvalue { i64, i64 } %v82, 0
  %v84 = extractvalue { i64, i64 } %v82, 1
  %v85 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v83, i64 %v84, i64 64) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v85, ptr %v36, align 8
  br label %bb50
bb14:
  %v86 = phi i64 [ %v275, %bb17 ], [ %v261, %bb50 ]
  %v87 = phi i64 [ %v276, %bb17 ], [ %v264, %bb50 ]
  %v88 = add i64 %v266, 1
  %v89 = icmp eq i64 %v88, 0
  %v90 = select i1 %v89, i8 0, i8 1
  %v91 = insertvalue { i8, { { i64 } } } undef, i8 %v90, 0
  %v92 = insertvalue { i8, { { i64 } } } %v91, i64 %v88, 1, 0, 0
  %v93 = extractvalue { i8, { { i64 } } } %v92, 0
  %v94 = zext i8 %v93 to i64
  %v95 = icmp eq i64 %v94, 1
  %v96 = extractvalue { i8, { { i64 } } } %v92, 1
  %v97 = alloca { { i64 } }, align 8
  store { { i64 } } %v96, ptr %v97, align 8
  %v98 = load i64, ptr %v97, align 8
  %v99 = icmp ugt i64 %v87, 0
  %v100 = xor i1 %v99, 1
  br i1 %v100, label %bb52, label %bb51
bb15:
  %v101 = extractvalue { i64, i64 } %v274, 1
  %v102 = mul i64 %v47, %v49
  %v103 = add i64 %v102, %v75
  %v104 = add i64 %v103, %v101
  %v105 = extractvalue { ptr, i64 } %v29, 1
  %v106 = icmp ult i64 %v104, %v105
  br i1 %v106, label %bb17, label %bb65
bb16:
  call void @llvm.nvvm.barrier0() #0
  br label %bb18
bb17:
  %v108 = extractvalue { ptr, i64 } %v29, 0
  %v109 = getelementptr inbounds i16, ptr %v108, i64 %v104
  %v110 = load i16, ptr %v109, align 2
  %v111 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_auto_round_tiled, i64 %v101
  %v112 = addrspacecast ptr addrspace(3) %v111 to ptr
  store i16 %v110, ptr %v112, align 2
  br label %bb14
bb18:
  %v113 = icmp ult i64 %v45, %v48
  %v114 = xor i1 %v113, 1
  br i1 %v114, label %bb40, label %bb19
bb19:
  %v115 = icmp eq i64 %v50, 0
  %v116 = xor i1 %v115, 1
  br i1 %v116, label %bb20, label %bb66
bb20:
  %v117 = udiv i64 %v75, %v50
  %v118 = icmp ne i32 %v34, 0
  %v119 = icmp eq i32 %v34, 0
  br i1 %v119, label %bb23, label %bb21
bb21:
  %v120 = mul i64 %v117, %v48
  %v121 = add i64 %v120, %v45
  %v122 = extractvalue { ptr, i64 } %v27, 1
  %v123 = icmp ult i64 %v121, %v122
  br i1 %v123, label %bb22, label %bb67
bb22:
  %v124 = extractvalue { ptr, i64 } %v27, 0
  %v125 = getelementptr inbounds i16, ptr %v124, i64 %v121
  %v126 = load i16, ptr %v125, align 2
  br label %bb25
bb23:
  %v127 = udiv i64 %v49, %v50
  %v128 = mul i64 %v45, %v127
  %v129 = add i64 %v128, %v117
  %v130 = extractvalue { ptr, i64 } %v27, 1
  %v131 = icmp ult i64 %v129, %v130
  br i1 %v131, label %bb24, label %bb68
bb24:
  %v132 = extractvalue { ptr, i64 } %v27, 0
  %v133 = getelementptr inbounds i16, ptr %v132, i64 %v129
  %v134 = load i16, ptr %v133, align 2
  br label %bb25
bb25:
  %v135 = phi i16 [ %v126, %bb22 ], [ %v134, %bb24 ]
  %v136 = call float @f16_to_f32(i16 %v135) #0
  br label %bb55
bb26:
  %v137 = add i64 %v48, 7
  %v138 = udiv i64 %v137, 8
  %v139 = mul i64 %v117, %v138
  %v140 = udiv i64 %v45, 8
  %v141 = add i64 %v139, %v140
  %v142 = urem i64 %v45, 8
  %v143 = mul i64 %v142, 4
  br label %bb28
bb27:
  %v144 = udiv i64 %v49, %v50
  %v145 = mul i64 %v45, %v144
  %v146 = add i64 %v145, %v117
  %v147 = udiv i64 %v146, 8
  %v148 = urem i64 %v146, 8
  %v149 = mul i64 %v148, 4
  br label %bb28
bb28:
  %v150 = phi i64 [ %v141, %bb26 ], [ %v147, %bb27 ]
  %v151 = phi i64 [ %v143, %bb26 ], [ %v149, %bb27 ]
  %v152 = extractvalue { ptr, i64 } %v28, 1
  %v153 = icmp ult i64 %v150, %v152
  br i1 %v153, label %bb29, label %bb69
bb29:
  %v154 = extractvalue { ptr, i64 } %v28, 0
  %v155 = getelementptr inbounds i32, ptr %v154, i64 %v150
  %v156 = load i32, ptr %v155, align 4
  %v157 = trunc i64 %v151 to i32
  %v158 = and i32 %v157, 31
  %v159 = lshr i32 %v156, %v158
  %v160 = and i32 %v159, 15
  %v161 = trunc i32 %v160 to i8
  %v162 = insertvalue { i64, i64 } undef, i64 0, 0
  %v163 = insertvalue { i64, i64 } %v162, i64 %v80, 1
  %v164 = extractvalue { i64, i64 } %v163, 0
  %v165 = extractvalue { i64, i64 } %v163, 1
  %v166 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v164, i64 %v165, i64 8) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v166, ptr %v37, align 8
  br label %bb56
bb30:
  %v167 = phi float [ %v201, %bb39 ], [ %v59, %bb56 ]
  %v168 = phi i64 [ %v298, %bb39 ], [ %v284, %bb56 ]
  %v169 = phi i64 [ %v299, %bb39 ], [ %v287, %bb56 ]
  %v170 = add i64 %v289, 1
  %v171 = icmp eq i64 %v170, 0
  %v172 = select i1 %v171, i8 0, i8 1
  %v173 = insertvalue { i8, { { i64 } } } undef, i8 %v172, 0
  %v174 = insertvalue { i8, { { i64 } } } %v173, i64 %v170, 1, 0, 0
  %v175 = extractvalue { i8, { { i64 } } } %v174, 0
  %v176 = zext i8 %v175 to i64
  %v177 = icmp eq i64 %v176, 1
  %v178 = extractvalue { i8, { { i64 } } } %v174, 1
  %v179 = alloca { { i64 } }, align 8
  store { { i64 } } %v178, ptr %v179, align 8
  %v180 = load i64, ptr %v179, align 8
  %v181 = icmp ugt i64 %v169, 0
  %v182 = xor i1 %v181, 1
  br i1 %v182, label %bb58, label %bb57
bb31:
  %v183 = extractvalue { i64, i64 } %v297, 1
  %v184 = xor i1 %v118, 1
  br i1 %v184, label %bb34, label %bb33
bb32:
  br label %bb40
bb33:
  %v185 = add i64 %v75, %v183
  %v186 = zext i32 3 to i64
  %v187 = and i64 %v186, 63
  %v188 = lshr i64 %v185, %v187
  %v189 = mul i64 %v188, %v48
  %v190 = add i64 %v189, %v45
  br label %bb35
bb34:
  %v191 = mul i64 %v45, %v49
  %v192 = add i64 %v191, %v75
  %v193 = add i64 %v192, %v183
  %v194 = udiv i64 %v193, 8
  br label %bb35
bb35:
  %v195 = phi i64 [ %v190, %bb33 ], [ %v194, %bb34 ]
  %v196 = extractvalue { ptr, i64 } %v26, 1
  %v197 = icmp ult i64 %v195, %v196
  br i1 %v197, label %bb36, label %bb70
bb36:
  %v198 = extractvalue { ptr, i64 } %v26, 0
  %v199 = getelementptr inbounds i32, ptr %v198, i64 %v195
  %v200 = load i32, ptr %v199, align 4
  br label %bb37
bb37:
  %v201 = phi float [ %v167, %bb36 ], [ %v226, %bb38 ]
  %v202 = phi i64 [ 0, %bb36 ], [ %v309, %bb38 ]
  %v203 = icmp ult i64 %v202, 8
  %v204 = xor i1 %v203, 1
  br i1 %v204, label %bb62, label %bb61
bb38:
  %v205 = extractvalue { i64, i64 } %v308, 1
  %v206 = mul i64 %v205, 4
  %v207 = trunc i64 %v206 to i32
  %v208 = and i32 %v207, 31
  %v209 = lshr i32 %v200, %v208
  %v210 = and i32 %v209, 15
  %v211 = trunc i32 %v210 to i8
  %v212 = sitofp i8 %v211 to float
  %v213 = sitofp i8 %v161 to float
  %v214 = fadd contract float %v213, 1.0
  %v215 = fsub contract float %v212, %v214
  %v216 = fmul contract float %v215, %v136
  %v217 = add i64 %v183, %v205
  %v218 = getelementptr inbounds i16, ptr addrspace(3) @__dynamic_smem_int4_gemm_auto_round_tiled, i64 %v217
  %v219 = addrspacecast ptr addrspace(3) %v218 to ptr
  %v220 = load i16, ptr %v219, align 2
  %v221 = zext i16 %v220 to i32
  %v222 = and i32 16, 31
  %v223 = shl i32 %v221, %v222
  %v224 = bitcast i32 %v223 to float
  %v225 = fmul contract float %v216, %v224
  %v226 = fadd contract float %v201, %v225
  br label %bb37
bb39:
  br label %bb30
bb40:
  %v227 = phi float [ %v59, %bb18 ], [ %v167, %bb32 ]
  call void @llvm.nvvm.barrier0() #0
  br label %bb41
bb41:
  br label %bb9
bb42:
  %v229 = bitcast float %v59 to i32
  %v230 = and i32 16, 31
  %v231 = lshr i32 %v229, %v230
  %v232 = trunc i32 %v231 to i16
  %v233 = mul i64 %v47, %v48
  %v234 = add i64 %v233, %v45
  %v235 = extractvalue { ptr, i64 } %v25, 0
  %v236 = getelementptr inbounds i16, ptr %v235, i64 %v234
  store i16 %v232, ptr %v236, align 2
  br label %bb43
bb43:
  br label %bb44
bb44:
  ret void
bb45:
  %v237 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v238 = getelementptr inbounds { i64, i64 }, ptr %v237, i32 0, i32 0
  %v239 = load i64, ptr %v238, align 8
  %v240 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v241 = getelementptr inbounds { i64, i64 }, ptr %v240, i32 0, i32 1
  %v242 = load i64, ptr %v241, align 8
  %v243 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 1
  %v244 = load i64, ptr %v243, align 8
  %v245 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 2
  %v246 = load i1, ptr %v245, align 1
  br label %bb9
bb46:
  %v247 = add i64 %v60, %v72
  %v248 = sub i64 %v61, 1
  %v249 = insertvalue { i64, i64 } undef, i64 1, 0
  %v250 = insertvalue { i64, i64 } %v249, i64 %v60, 1
  br label %bb48
bb47:
  %v251 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb48
bb48:
  %v252 = phi { i64, i64 } [ %v250, %bb46 ], [ %v251, %bb47 ]
  %v253 = phi i64 [ %v247, %bb46 ], [ %v60, %bb47 ]
  %v254 = phi i64 [ %v248, %bb46 ], [ %v61, %bb47 ]
  %v255 = extractvalue { i64, i64 } %v252, 0
  %v256 = bitcast i64 %v255 to i64
  %v257 = icmp eq i64 %v256, 0
  br i1 %v257, label %bb12, label %bb49
bb49:
  %v258 = icmp eq i64 %v256, 1
  br i1 %v258, label %bb11, label %bb10
bb50:
  %v259 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 0
  %v260 = getelementptr inbounds { i64, i64 }, ptr %v259, i32 0, i32 0
  %v261 = load i64, ptr %v260, align 8
  %v262 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 0
  %v263 = getelementptr inbounds { i64, i64 }, ptr %v262, i32 0, i32 1
  %v264 = load i64, ptr %v263, align 8
  %v265 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 1
  %v266 = load i64, ptr %v265, align 8
  %v267 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 2
  %v268 = load i1, ptr %v267, align 1
  br label %bb14
bb51:
  %v269 = add i64 %v86, %v98
  %v270 = sub i64 %v87, 1
  %v271 = insertvalue { i64, i64 } undef, i64 1, 0
  %v272 = insertvalue { i64, i64 } %v271, i64 %v86, 1
  br label %bb53
bb52:
  %v273 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb53
bb53:
  %v274 = phi { i64, i64 } [ %v272, %bb51 ], [ %v273, %bb52 ]
  %v275 = phi i64 [ %v269, %bb51 ], [ %v86, %bb52 ]
  %v276 = phi i64 [ %v270, %bb51 ], [ %v87, %bb52 ]
  %v277 = extractvalue { i64, i64 } %v274, 0
  %v278 = bitcast i64 %v277 to i64
  %v279 = icmp eq i64 %v278, 0
  br i1 %v279, label %bb16, label %bb54
bb54:
  %v280 = icmp eq i64 %v278, 1
  br i1 %v280, label %bb15, label %bb10
bb55:
  %v281 = xor i1 %v118, 1
  br i1 %v281, label %bb27, label %bb26
bb56:
  %v282 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 0
  %v283 = getelementptr inbounds { i64, i64 }, ptr %v282, i32 0, i32 0
  %v284 = load i64, ptr %v283, align 8
  %v285 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 0
  %v286 = getelementptr inbounds { i64, i64 }, ptr %v285, i32 0, i32 1
  %v287 = load i64, ptr %v286, align 8
  %v288 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 1
  %v289 = load i64, ptr %v288, align 8
  %v290 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v37, i32 0, i32 2
  %v291 = load i1, ptr %v290, align 1
  br label %bb30
bb57:
  %v292 = add i64 %v168, %v180
  %v293 = sub i64 %v169, 1
  %v294 = insertvalue { i64, i64 } undef, i64 1, 0
  %v295 = insertvalue { i64, i64 } %v294, i64 %v168, 1
  br label %bb59
bb58:
  %v296 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb59
bb59:
  %v297 = phi { i64, i64 } [ %v295, %bb57 ], [ %v296, %bb58 ]
  %v298 = phi i64 [ %v292, %bb57 ], [ %v168, %bb58 ]
  %v299 = phi i64 [ %v293, %bb57 ], [ %v169, %bb58 ]
  %v300 = extractvalue { i64, i64 } %v297, 0
  %v301 = bitcast i64 %v300 to i64
  %v302 = icmp eq i64 %v301, 0
  br i1 %v302, label %bb32, label %bb60
bb60:
  %v303 = icmp eq i64 %v301, 1
  br i1 %v303, label %bb31, label %bb10
bb61:
  %v304 = add i64 %v202, 1
  %v305 = insertvalue { i64, i64 } undef, i64 1, 0
  %v306 = insertvalue { i64, i64 } %v305, i64 %v202, 1
  br label %bb63
bb62:
  %v307 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb63
bb63:
  %v308 = phi { i64, i64 } [ %v306, %bb61 ], [ %v307, %bb62 ]
  %v309 = phi i64 [ %v304, %bb61 ], [ %v202, %bb62 ]
  %v310 = extractvalue { i64, i64 } %v308, 0
  %v311 = bitcast i64 %v310 to i64
  %v312 = icmp eq i64 %v311, 0
  br i1 %v312, label %bb39, label %bb64
bb64:
  %v313 = icmp eq i64 %v311, 1
  br i1 %v313, label %bb38, label %bb10
bb65:
  unreachable
bb66:
  unreachable
bb67:
  unreachable
bb68:
  unreachable
bb69:
  unreachable
bb70:
  unreachable
}

define void @reduce_partial_sums_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi { ptr, i64 } [ %v9, %entry ]
  %v12 = phi i32 [ %v4, %entry ]
  %v13 = phi i32 [ %v5, %entry ]
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v15 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v16 = mul i32 %v15, 64
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v18 = add i32 %v16, %v17
  %v19 = zext i32 %v18 to i64
  %v20 = zext i32 %v12 to i64
  %v21 = icmp uge i64 %v19, %v20
  %v22 = xor i1 %v21, 1
  br i1 %v22, label %bb5, label %bb4
bb4:
  br label %bb11
bb5:
  %v23 = zext i32 %v13 to i64
  br label %bb6
bb6:
  %v24 = phi float [ 0.0, %bb5 ], [ %v42, %bb10 ]
  %v25 = phi i64 [ 0, %bb5 ], [ %v48, %bb10 ]
  %v26 = icmp ult i64 %v25, %v23
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb13, label %bb12
bb7:
  unreachable
bb8:
  %v28 = extractvalue { i64, i64 } %v47, 1
  %v29 = mul i64 %v28, %v20
  %v30 = add i64 %v29, %v19
  %v31 = extractvalue { ptr, i64 } %v11, 1
  %v32 = icmp ult i64 %v30, %v31
  br i1 %v32, label %bb10, label %bb16
bb9:
  %v33 = bitcast float %v24 to i32
  %v34 = and i32 16, 31
  %v35 = lshr i32 %v33, %v34
  %v36 = trunc i32 %v35 to i16
  %v37 = extractvalue { ptr, i64 } %v10, 0
  %v38 = getelementptr inbounds i16, ptr %v37, i64 %v19
  store i16 %v36, ptr %v38, align 2
  br label %bb11
bb10:
  %v39 = extractvalue { ptr, i64 } %v11, 0
  %v40 = getelementptr inbounds float, ptr %v39, i64 %v30
  %v41 = load float, ptr %v40, align 4
  %v42 = fadd contract float %v24, %v41
  br label %bb6
bb11:
  ret void
bb12:
  %v43 = add i64 %v25, 1
  %v44 = insertvalue { i64, i64 } undef, i64 1, 0
  %v45 = insertvalue { i64, i64 } %v44, i64 %v25, 1
  br label %bb14
bb13:
  %v46 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb14
bb14:
  %v47 = phi { i64, i64 } [ %v45, %bb12 ], [ %v46, %bb13 ]
  %v48 = phi i64 [ %v43, %bb12 ], [ %v25, %bb13 ]
  %v49 = extractvalue { i64, i64 } %v47, 0
  %v50 = bitcast i64 %v49 to i64
  %v51 = icmp eq i64 %v50, 0
  br i1 %v51, label %bb9, label %bb15
bb15:
  %v52 = icmp eq i64 %v50, 1
  br i1 %v52, label %bb8, label %bb7
bb16:
  unreachable
}

define void @int4_gemm_warp(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v7, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v9, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v15, %entry ]
  %v25 = phi { ptr, i64 } [ %v17, %entry ]
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi { ptr, i64 } [ %v23, %entry ]
  %v29 = phi i32 [ %v10, %entry ]
  %v30 = phi i32 [ %v11, %entry ]
  %v31 = phi i32 [ %v12, %entry ]
  %v32 = phi i32 [ %v13, %entry ]
  %v33 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v34 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb2
bb2:
  %v37 = zext i32 %v36 to i64
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb3
bb3:
  %v39 = zext i32 %v38 to i64
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb4
bb4:
  %v41 = mul i32 %v40, 8
  %v42 = trunc i64 %v39 to i32
  %v43 = add i32 %v41, %v42
  %v44 = zext i32 %v43 to i64
  %v45 = zext i32 %v29 to i64
  %v46 = zext i32 %v30 to i64
  %v47 = zext i32 %v31 to i64
  %v48 = icmp eq i64 %v47, 0
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb5, label %bb55
bb5:
  %v50 = udiv i64 %v46, %v47
  %v51 = icmp uge i64 %v44, %v45
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb7, label %bb6
bb6:
  br label %bb34
bb7:
  %v53 = insertvalue { i64, i64 } undef, i64 %v37, 0
  %v54 = insertvalue { i64, i64 } %v53, i64 %v50, 1
  %v55 = extractvalue { i64, i64 } %v54, 0
  %v56 = extractvalue { i64, i64 } %v54, 1
  %v57 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v55, i64 %v56, i64 32) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v57, ptr %v33, align 8
  br label %bb35
bb8:
  %v58 = phi float [ %v124, %bb23 ], [ 0.0, %bb35 ]
  %v59 = phi i64 [ %v207, %bb23 ], [ %v193, %bb35 ]
  %v60 = phi i64 [ %v208, %bb23 ], [ %v196, %bb35 ]
  %v61 = add i64 %v198, 1
  %v62 = icmp eq i64 %v61, 0
  %v63 = select i1 %v62, i8 0, i8 1
  %v64 = insertvalue { i8, { { i64 } } } undef, i8 %v63, 0
  %v65 = insertvalue { i8, { { i64 } } } %v64, i64 %v61, 1, 0, 0
  %v66 = extractvalue { i8, { { i64 } } } %v65, 0
  %v67 = zext i8 %v66 to i64
  %v68 = icmp eq i64 %v67, 1
  %v69 = extractvalue { i8, { { i64 } } } %v65, 1
  %v70 = alloca { { i64 } }, align 8
  store { { i64 } } %v69, ptr %v70, align 8
  %v71 = load i64, ptr %v70, align 8
  %v72 = icmp ugt i64 %v60, 0
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb37, label %bb36
bb9:
  unreachable
bb10:
  %v74 = extractvalue { i64, i64 } %v206, 1
  %v75 = mul i64 %v74, %v47
  %v76 = icmp ne i32 %v32, 0
  %v77 = icmp eq i32 %v32, 0
  br i1 %v77, label %bb14, label %bb12
bb11:
  %v78 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v58, i32 16, i32 31) #0
  br label %bb40
bb12:
  %v79 = mul i64 %v74, %v45
  %v80 = add i64 %v79, %v44
  %v81 = extractvalue { ptr, i64 } %v26, 1
  %v82 = icmp ult i64 %v80, %v81
  br i1 %v82, label %bb13, label %bb56
bb13:
  %v83 = extractvalue { ptr, i64 } %v26, 0
  %v84 = getelementptr inbounds i16, ptr %v83, i64 %v80
  %v85 = load i16, ptr %v84, align 2
  br label %bb16
bb14:
  %v86 = mul i64 %v44, %v50
  %v87 = add i64 %v86, %v74
  %v88 = extractvalue { ptr, i64 } %v26, 1
  %v89 = icmp ult i64 %v87, %v88
  br i1 %v89, label %bb15, label %bb57
bb15:
  %v90 = extractvalue { ptr, i64 } %v26, 0
  %v91 = getelementptr inbounds i16, ptr %v90, i64 %v87
  %v92 = load i16, ptr %v91, align 2
  br label %bb16
bb16:
  %v93 = phi i16 [ %v85, %bb13 ], [ %v92, %bb15 ]
  %v94 = call float @f16_to_f32(i16 %v93) #0
  br label %bb41
bb17:
  %v95 = add i64 %v45, 7
  %v96 = udiv i64 %v95, 8
  %v97 = mul i64 %v74, %v96
  %v98 = udiv i64 %v44, 8
  %v99 = add i64 %v97, %v98
  %v100 = urem i64 %v44, 8
  %v101 = mul i64 %v100, 4
  br label %bb19
bb18:
  %v102 = mul i64 %v44, %v50
  %v103 = add i64 %v102, %v74
  %v104 = udiv i64 %v103, 8
  %v105 = urem i64 %v103, 8
  %v106 = mul i64 %v105, 4
  br label %bb19
bb19:
  %v107 = phi i64 [ %v99, %bb17 ], [ %v104, %bb18 ]
  %v108 = phi i64 [ %v101, %bb17 ], [ %v106, %bb18 ]
  %v109 = extractvalue { ptr, i64 } %v27, 1
  %v110 = icmp ult i64 %v107, %v109
  br i1 %v110, label %bb20, label %bb58
bb20:
  %v111 = extractvalue { ptr, i64 } %v27, 0
  %v112 = getelementptr inbounds i32, ptr %v111, i64 %v107
  %v113 = load i32, ptr %v112, align 4
  %v114 = trunc i64 %v108 to i32
  %v115 = and i32 %v114, 31
  %v116 = lshr i32 %v113, %v115
  %v117 = and i32 %v116, 15
  %v118 = trunc i32 %v117 to i8
  %v119 = insertvalue { i64, i64 } undef, i64 0, 0
  %v120 = insertvalue { i64, i64 } %v119, i64 %v47, 1
  %v121 = extractvalue { i64, i64 } %v120, 0
  %v122 = extractvalue { i64, i64 } %v120, 1
  %v123 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v121, i64 %v122, i64 8) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v123, ptr %v34, align 8
  br label %bb42
bb21:
  %v124 = phi float [ %v157, %bb30 ], [ %v58, %bb42 ]
  %v125 = phi i64 [ %v232, %bb30 ], [ %v218, %bb42 ]
  %v126 = phi i64 [ %v233, %bb30 ], [ %v221, %bb42 ]
  %v127 = add i64 %v223, 1
  %v128 = icmp eq i64 %v127, 0
  %v129 = select i1 %v128, i8 0, i8 1
  %v130 = insertvalue { i8, { { i64 } } } undef, i8 %v129, 0
  %v131 = insertvalue { i8, { { i64 } } } %v130, i64 %v127, 1, 0, 0
  %v132 = extractvalue { i8, { { i64 } } } %v131, 0
  %v133 = zext i8 %v132 to i64
  %v134 = icmp eq i64 %v133, 1
  %v135 = extractvalue { i8, { { i64 } } } %v131, 1
  %v136 = alloca { { i64 } }, align 8
  store { { i64 } } %v135, ptr %v136, align 8
  %v137 = load i64, ptr %v136, align 8
  %v138 = icmp ugt i64 %v126, 0
  %v139 = xor i1 %v138, 1
  br i1 %v139, label %bb44, label %bb43
bb22:
  %v140 = extractvalue { i64, i64 } %v231, 1
  %v141 = add i64 %v75, %v140
  %v142 = xor i1 %v76, 1
  br i1 %v142, label %bb25, label %bb24
bb23:
  br label %bb8
bb24:
  %v143 = zext i32 3 to i64
  %v144 = and i64 %v143, 63
  %v145 = lshr i64 %v141, %v144
  %v146 = mul i64 %v145, %v45
  %v147 = add i64 %v146, %v44
  br label %bb26
bb25:
  %v148 = mul i64 %v44, %v46
  %v149 = add i64 %v148, %v141
  %v150 = udiv i64 %v149, 8
  br label %bb26
bb26:
  %v151 = phi i64 [ %v147, %bb24 ], [ %v150, %bb25 ]
  %v152 = extractvalue { ptr, i64 } %v25, 1
  %v153 = icmp ult i64 %v151, %v152
  br i1 %v153, label %bb27, label %bb59
bb27:
  %v154 = extractvalue { ptr, i64 } %v25, 0
  %v155 = getelementptr inbounds i32, ptr %v154, i64 %v151
  %v156 = load i32, ptr %v155, align 4
  br label %bb28
bb28:
  %v157 = phi float [ %v124, %bb27 ], [ %v184, %bb31 ]
  %v158 = phi i64 [ 0, %bb27 ], [ %v243, %bb31 ]
  %v159 = icmp ult i64 %v158, 8
  %v160 = xor i1 %v159, 1
  br i1 %v160, label %bb48, label %bb47
bb29:
  %v161 = extractvalue { i64, i64 } %v242, 1
  %v162 = mul i64 %v161, 4
  %v163 = trunc i64 %v162 to i32
  %v164 = and i32 %v163, 31
  %v165 = lshr i32 %v156, %v164
  %v166 = and i32 %v165, 15
  %v167 = trunc i32 %v166 to i8
  %v168 = sitofp i8 %v167 to float
  %v169 = sitofp i8 %v118 to float
  %v170 = fadd contract float %v169, 1.0
  %v171 = fsub contract float %v168, %v170
  %v172 = fmul contract float %v171, %v94
  %v173 = add i64 %v141, %v161
  %v174 = extractvalue { ptr, i64 } %v28, 1
  %v175 = icmp ult i64 %v173, %v174
  br i1 %v175, label %bb31, label %bb60
bb30:
  br label %bb21
bb31:
  %v176 = extractvalue { ptr, i64 } %v28, 0
  %v177 = getelementptr inbounds i16, ptr %v176, i64 %v173
  %v178 = load i16, ptr %v177, align 2
  %v179 = zext i16 %v178 to i32
  %v180 = and i32 16, 31
  %v181 = shl i32 %v179, %v180
  %v182 = bitcast i32 %v181 to float
  %v183 = fmul contract float %v172, %v182
  %v184 = fadd contract float %v157, %v183
  br label %bb28
bb32:
  %v185 = bitcast float %v254 to i32
  %v186 = and i32 16, 31
  %v187 = lshr i32 %v185, %v186
  %v188 = trunc i32 %v187 to i16
  %v189 = extractvalue { ptr, i64 } %v24, 0
  %v190 = getelementptr inbounds i16, ptr %v189, i64 %v44
  store i16 %v188, ptr %v190, align 2
  br label %bb33
bb33:
  br label %bb34
bb34:
  ret void
bb35:
  %v191 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 0
  %v192 = getelementptr inbounds { i64, i64 }, ptr %v191, i32 0, i32 0
  %v193 = load i64, ptr %v192, align 8
  %v194 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 0
  %v195 = getelementptr inbounds { i64, i64 }, ptr %v194, i32 0, i32 1
  %v196 = load i64, ptr %v195, align 8
  %v197 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 1
  %v198 = load i64, ptr %v197, align 8
  %v199 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 2
  %v200 = load i1, ptr %v199, align 1
  br label %bb8
bb36:
  %v201 = add i64 %v59, %v71
  %v202 = sub i64 %v60, 1
  %v203 = insertvalue { i64, i64 } undef, i64 1, 0
  %v204 = insertvalue { i64, i64 } %v203, i64 %v59, 1
  br label %bb38
bb37:
  %v205 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb38
bb38:
  %v206 = phi { i64, i64 } [ %v204, %bb36 ], [ %v205, %bb37 ]
  %v207 = phi i64 [ %v201, %bb36 ], [ %v59, %bb37 ]
  %v208 = phi i64 [ %v202, %bb36 ], [ %v60, %bb37 ]
  %v209 = extractvalue { i64, i64 } %v206, 0
  %v210 = bitcast i64 %v209 to i64
  %v211 = icmp eq i64 %v210, 0
  br i1 %v211, label %bb11, label %bb39
bb39:
  %v212 = icmp eq i64 %v210, 1
  br i1 %v212, label %bb10, label %bb9
bb40:
  %v213 = fadd contract float %v58, %v78
  %v214 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v213, i32 8, i32 31) #0
  br label %bb51
bb41:
  %v215 = xor i1 %v76, 1
  br i1 %v215, label %bb18, label %bb17
bb42:
  %v216 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v217 = getelementptr inbounds { i64, i64 }, ptr %v216, i32 0, i32 0
  %v218 = load i64, ptr %v217, align 8
  %v219 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v220 = getelementptr inbounds { i64, i64 }, ptr %v219, i32 0, i32 1
  %v221 = load i64, ptr %v220, align 8
  %v222 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 1
  %v223 = load i64, ptr %v222, align 8
  %v224 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 2
  %v225 = load i1, ptr %v224, align 1
  br label %bb21
bb43:
  %v226 = add i64 %v125, %v137
  %v227 = sub i64 %v126, 1
  %v228 = insertvalue { i64, i64 } undef, i64 1, 0
  %v229 = insertvalue { i64, i64 } %v228, i64 %v125, 1
  br label %bb45
bb44:
  %v230 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb45
bb45:
  %v231 = phi { i64, i64 } [ %v229, %bb43 ], [ %v230, %bb44 ]
  %v232 = phi i64 [ %v226, %bb43 ], [ %v125, %bb44 ]
  %v233 = phi i64 [ %v227, %bb43 ], [ %v126, %bb44 ]
  %v234 = extractvalue { i64, i64 } %v231, 0
  %v235 = bitcast i64 %v234 to i64
  %v236 = icmp eq i64 %v235, 0
  br i1 %v236, label %bb23, label %bb46
bb46:
  %v237 = icmp eq i64 %v235, 1
  br i1 %v237, label %bb22, label %bb9
bb47:
  %v238 = add i64 %v158, 1
  %v239 = insertvalue { i64, i64 } undef, i64 1, 0
  %v240 = insertvalue { i64, i64 } %v239, i64 %v158, 1
  br label %bb49
bb48:
  %v241 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb49
bb49:
  %v242 = phi { i64, i64 } [ %v240, %bb47 ], [ %v241, %bb48 ]
  %v243 = phi i64 [ %v238, %bb47 ], [ %v158, %bb48 ]
  %v244 = extractvalue { i64, i64 } %v242, 0
  %v245 = bitcast i64 %v244 to i64
  %v246 = icmp eq i64 %v245, 0
  br i1 %v246, label %bb30, label %bb50
bb50:
  %v247 = icmp eq i64 %v245, 1
  br i1 %v247, label %bb29, label %bb9
bb51:
  %v248 = fadd contract float %v213, %v214
  %v249 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v248, i32 4, i32 31) #0
  br label %bb52
bb52:
  %v250 = fadd contract float %v248, %v249
  %v251 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v250, i32 2, i32 31) #0
  br label %bb53
bb53:
  %v252 = fadd contract float %v250, %v251
  %v253 = call float @llvm.nvvm.shfl.sync.bfly.f32(i32 4294967295, float %v252, i32 1, i32 31) #0
  br label %bb54
bb54:
  %v254 = fadd contract float %v252, %v253
  %v255 = icmp eq i64 %v37, 0
  br i1 %v255, label %bb32, label %bb33
bb55:
  unreachable
bb56:
  unreachable
bb57:
  unreachable
bb58:
  unreachable
bb59:
  unreachable
bb60:
  unreachable
}

define void @int4_gemm_gguf(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v7, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v9, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v16, %entry ]
  %v26 = phi { ptr, i64 } [ %v18, %entry ]
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = alloca { ptr, i64 }, align 8
  store { ptr, i64 } %v25, ptr %v35, align 8
  %v36 = bitcast i32 %v30 to i32
  %v37 = bitcast i32 %v31 to i32
  %v38 = bitcast i32 %v32 to i32
  %v39 = bitcast i32 %v33 to i32
  %v40 = bitcast i32 %v34 to i32
  %v41 = extractvalue { ptr, i64 } %v26, 0
  %v42 = extractvalue { ptr, i64 } %v26, 1
  %v43 = extractvalue { ptr, i64 } %v27, 0
  %v44 = extractvalue { ptr, i64 } %v27, 1
  %v45 = extractvalue { ptr, i64 } %v28, 0
  %v46 = extractvalue { ptr, i64 } %v28, 1
  %v47 = extractvalue { ptr, i64 } %v29, 0
  %v48 = extractvalue { ptr, i64 } %v29, 1
  call void @int4_gemm_innerNtB2_4GgufEB4_(ptr %v35, ptr %v41, i64 %v42, ptr %v43, i64 %v44, ptr %v45, i64 %v46, ptr %v47, i64 %v48, i32 %v36, i32 %v37, i32 %v38, i32 %v39, i32 %v40) #0
  br label %bb1
bb1:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()

define void @int4_dequant_to_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10) #0 {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v1, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v3, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v5, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v7, 1
  br label %bb0
bb0:
  %v19 = phi { ptr, i64 } [ %v12, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = phi { ptr, i64 } [ %v18, %entry ]
  %v23 = phi i32 [ %v8, %entry ]
  %v24 = phi i32 [ %v9, %entry ]
  %v25 = phi i32 [ %v10, %entry ]
  %v26 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v28 = mul i32 %v26, %v27
  %v29 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v30 = add i32 %v28, %v29
  %v31 = zext i32 %v30 to i64
  %v32 = zext i32 %v23 to i64
  %v33 = icmp uge i64 %v31, %v32
  %v34 = xor i1 %v33, 1
  br i1 %v34, label %bb5, label %bb4
bb4:
  br label %bb20
bb5:
  %v35 = icmp eq i32 %v25, 0
  %v36 = xor i1 %v35, 1
  br i1 %v36, label %bb6, label %bb34
bb6:
  %v37 = udiv i32 %v24, %v25
  %v38 = zext i32 %v37 to i64
  %v39 = zext i32 %v24 to i64
  %v40 = zext i32 %v25 to i64
  br label %bb7
bb7:
  %v41 = phi i64 [ 0, %bb6 ], [ %v106, %bb15 ]
  %v42 = icmp ult i64 %v41, %v38
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb22, label %bb21
bb8:
  unreachable
bb9:
  %v44 = extractvalue { i64, i64 } %v105, 1
  %v45 = mul i64 %v44, %v32
  %v46 = add i64 %v45, %v31
  %v47 = extractvalue { ptr, i64 } %v21, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb11, label %bb35
bb10:
  br label %bb20
bb11:
  %v49 = extractvalue { ptr, i64 } %v21, 0
  %v50 = getelementptr inbounds i16, ptr %v49, i64 %v46
  %v51 = load i16, ptr %v50, align 2
  %v52 = call float @f16_to_f32(i16 %v51) #0
  br label %bb25
bb12:
  %v53 = extractvalue { ptr, i64 } %v22, 0
  %v54 = getelementptr inbounds i32, ptr %v53, i64 %v114
  %v55 = load i32, ptr %v54, align 4
  %v56 = trunc i64 %v116 to i32
  %v57 = and i32 %v56, 31
  %v58 = lshr i32 %v55, %v57
  %v59 = and i32 %v58, 15
  %v60 = trunc i32 %v59 to i8
  %v61 = udiv i64 %v40, 8
  br label %bb13
bb13:
  %v62 = phi i64 [ 0, %bb12 ], [ %v124, %bb19 ]
  %v63 = icmp ult i64 %v62, %v61
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb27, label %bb26
bb14:
  %v65 = extractvalue { i64, i64 } %v123, 1
  %v66 = mul i64 %v44, %v61
  %v67 = add i64 %v66, %v65
  %v68 = mul i64 %v67, %v32
  %v69 = add i64 %v68, %v31
  %v70 = extractvalue { ptr, i64 } %v20, 1
  %v71 = icmp ult i64 %v69, %v70
  br i1 %v71, label %bb16, label %bb36
bb15:
  br label %bb7
bb16:
  %v72 = extractvalue { ptr, i64 } %v20, 0
  %v73 = getelementptr inbounds i32, ptr %v72, i64 %v69
  %v74 = load i32, ptr %v73, align 4
  br label %bb17
bb17:
  %v75 = phi i32 [ 0, %bb16 ], [ %v134, %bb18 ]
  %v76 = icmp ult i32 %v75, 8
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb31, label %bb30
bb18:
  %v78 = extractvalue { i32, i32 } %v133, 1
  %v79 = mul i32 %v78, 4
  %v80 = and i32 %v79, 31
  %v81 = lshr i32 %v74, %v80
  %v82 = and i32 %v81, 15
  %v83 = trunc i32 %v82 to i8
  %v84 = add i8 %v60, 1
  %v85 = sub i8 %v83, %v84
  %v86 = sitofp i8 %v85 to float
  %v87 = fmul contract float %v86, %v52
  %v88 = bitcast float %v87 to i32
  %v89 = and i32 16, 31
  %v90 = lshr i32 %v88, %v89
  %v91 = trunc i32 %v90 to i16
  %v92 = mul i64 %v31, %v39
  %v93 = mul i64 %v44, %v40
  %v94 = add i64 %v92, %v93
  %v95 = mul i64 %v65, 8
  %v96 = add i64 %v94, %v95
  %v97 = zext i32 %v78 to i64
  %v98 = add i64 %v96, %v97
  %v99 = extractvalue { ptr, i64 } %v19, 0
  %v100 = getelementptr inbounds i16, ptr %v99, i64 %v98
  store i16 %v91, ptr %v100, align 2
  br label %bb17
bb19:
  br label %bb13
bb20:
  ret void
bb21:
  %v101 = add i64 %v41, 1
  %v102 = insertvalue { i64, i64 } undef, i64 1, 0
  %v103 = insertvalue { i64, i64 } %v102, i64 %v41, 1
  br label %bb23
bb22:
  %v104 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb23
bb23:
  %v105 = phi { i64, i64 } [ %v103, %bb21 ], [ %v104, %bb22 ]
  %v106 = phi i64 [ %v101, %bb21 ], [ %v41, %bb22 ]
  %v107 = extractvalue { i64, i64 } %v105, 0
  %v108 = bitcast i64 %v107 to i64
  %v109 = icmp eq i64 %v108, 0
  br i1 %v109, label %bb10, label %bb24
bb24:
  %v110 = icmp eq i64 %v108, 1
  br i1 %v110, label %bb9, label %bb8
bb25:
  %v111 = udiv i64 %v32, 8
  %v112 = mul i64 %v44, %v111
  %v113 = udiv i64 %v31, 8
  %v114 = add i64 %v112, %v113
  %v115 = urem i64 %v31, 8
  %v116 = mul i64 %v115, 4
  %v117 = extractvalue { ptr, i64 } %v22, 1
  %v118 = icmp ult i64 %v114, %v117
  br i1 %v118, label %bb12, label %bb37
bb26:
  %v119 = add i64 %v62, 1
  %v120 = insertvalue { i64, i64 } undef, i64 1, 0
  %v121 = insertvalue { i64, i64 } %v120, i64 %v62, 1
  br label %bb28
bb27:
  %v122 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb28
bb28:
  %v123 = phi { i64, i64 } [ %v121, %bb26 ], [ %v122, %bb27 ]
  %v124 = phi i64 [ %v119, %bb26 ], [ %v62, %bb27 ]
  %v125 = extractvalue { i64, i64 } %v123, 0
  %v126 = bitcast i64 %v125 to i64
  %v127 = icmp eq i64 %v126, 0
  br i1 %v127, label %bb15, label %bb29
bb29:
  %v128 = icmp eq i64 %v126, 1
  br i1 %v128, label %bb14, label %bb8
bb30:
  %v129 = add i32 %v75, 1
  %v130 = insertvalue { i32, i32 } undef, i32 1, 0
  %v131 = insertvalue { i32, i32 } %v130, i32 %v75, 1
  br label %bb32
bb31:
  %v132 = insertvalue { i32, i32 } undef, i32 0, 0
  br label %bb32
bb32:
  %v133 = phi { i32, i32 } [ %v131, %bb30 ], [ %v132, %bb31 ]
  %v134 = phi i32 [ %v129, %bb30 ], [ %v75, %bb31 ]
  %v135 = extractvalue { i32, i32 } %v133, 0
  %v136 = zext i32 %v135 to i64
  %v137 = icmp eq i64 %v136, 0
  br i1 %v137, label %bb19, label %bb33
bb33:
  %v138 = icmp eq i64 %v136, 1
  br i1 %v138, label %bb18, label %bb8
bb34:
  unreachable
bb35:
  unreachable
bb36:
  unreachable
bb37:
  unreachable
}

define void @int4_gemm_auto_round_ksplit(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13, i32 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v7, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v9, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v16, %entry ]
  %v26 = phi { ptr, i64 } [ %v18, %entry ]
  %v27 = phi { ptr, i64 } [ %v20, %entry ]
  %v28 = phi { ptr, i64 } [ %v22, %entry ]
  %v29 = phi { ptr, i64 } [ %v24, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi i32 [ %v13, %entry ]
  %v34 = phi i32 [ %v14, %entry ]
  %v35 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v36 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v39 = mul i32 %v38, 64
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v41 = add i32 %v39, %v40
  %v42 = zext i32 %v41 to i64
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb4
bb4:
  %v44 = zext i32 %v43 to i64
  %v45 = zext i32 %v30 to i64
  %v46 = zext i32 %v31 to i64
  %v47 = zext i32 %v32 to i64
  %v48 = icmp uge i64 %v42, %v45
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb6, label %bb5
bb5:
  br label %bb44
bb6:
  %v50 = zext i32 %v34 to i64
  %v51 = add i64 %v46, %v50
  %v52 = sub i64 %v51, 1
  %v53 = icmp eq i64 %v50, 0
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb7, label %bb60
bb7:
  %v55 = udiv i64 %v52, %v50
  %v56 = mul i64 %v44, %v55
  %v57 = add i64 %v56, %v55
  %v58 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v57, i64 %v46) #0
  br label %bb8
bb8:
  %v59 = icmp eq i64 %v47, 0
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb9, label %bb61
bb9:
  %v61 = udiv i64 %v56, %v47
  %v62 = mul i64 %v61, %v47
  %v63 = add i64 %v58, %v47
  %v64 = sub i64 %v63, 1
  %v65 = udiv i64 %v64, %v47
  %v66 = mul i64 %v65, %v47
  %v67 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v66, i64 %v46) #0
  br label %bb10
bb10:
  %v68 = insertvalue { i64, i64 } undef, i64 %v62, 0
  %v69 = insertvalue { i64, i64 } %v68, i64 %v67, 1
  %v70 = extractvalue { i64, i64 } %v69, 0
  %v71 = extractvalue { i64, i64 } %v69, 1
  %v72 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v70, i64 %v71, i64 %v47) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v72, ptr %v35, align 8
  br label %bb45
bb11:
  %v73 = phi float [ %v144, %bb42 ], [ 0.0, %bb45 ]
  %v74 = phi i64 [ %v232, %bb42 ], [ %v218, %bb45 ]
  %v75 = phi i64 [ %v233, %bb42 ], [ %v221, %bb45 ]
  %v76 = add i64 %v223, 1
  %v77 = icmp eq i64 %v76, 0
  %v78 = select i1 %v77, i8 0, i8 1
  %v79 = insertvalue { i8, { { i64 } } } undef, i8 %v78, 0
  %v80 = insertvalue { i8, { { i64 } } } %v79, i64 %v76, 1, 0, 0
  %v81 = extractvalue { i8, { { i64 } } } %v80, 0
  %v82 = zext i8 %v81 to i64
  %v83 = icmp eq i64 %v82, 1
  %v84 = extractvalue { i8, { { i64 } } } %v80, 1
  %v85 = alloca { { i64 } }, align 8
  store { { i64 } } %v84, ptr %v85, align 8
  %v86 = load i64, ptr %v85, align 8
  %v87 = icmp ugt i64 %v75, 0
  %v88 = xor i1 %v87, 1
  br i1 %v88, label %bb47, label %bb46
bb12:
  unreachable
bb13:
  %v89 = extractvalue { i64, i64 } %v231, 1
  %v90 = udiv i64 %v89, %v47
  %v91 = icmp ne i32 %v33, 0
  %v92 = icmp eq i32 %v33, 0
  br i1 %v92, label %bb17, label %bb15
bb14:
  %v93 = mul i64 %v44, %v45
  %v94 = add i64 %v93, %v42
  %v95 = extractvalue { ptr, i64 } %v25, 1
  %v96 = icmp ult i64 %v94, %v95
  br i1 %v96, label %bb43, label %bb62
bb15:
  %v97 = mul i64 %v90, %v45
  %v98 = add i64 %v97, %v42
  %v99 = extractvalue { ptr, i64 } %v27, 1
  %v100 = icmp ult i64 %v98, %v99
  br i1 %v100, label %bb16, label %bb63
bb16:
  %v101 = extractvalue { ptr, i64 } %v27, 0
  %v102 = getelementptr inbounds i16, ptr %v101, i64 %v98
  %v103 = load i16, ptr %v102, align 2
  br label %bb19
bb17:
  %v104 = udiv i64 %v46, %v47
  %v105 = mul i64 %v42, %v104
  %v106 = add i64 %v105, %v90
  %v107 = extractvalue { ptr, i64 } %v27, 1
  %v108 = icmp ult i64 %v106, %v107
  br i1 %v108, label %bb18, label %bb64
bb18:
  %v109 = extractvalue { ptr, i64 } %v27, 0
  %v110 = getelementptr inbounds i16, ptr %v109, i64 %v106
  %v111 = load i16, ptr %v110, align 2
  br label %bb19
bb19:
  %v112 = phi i16 [ %v103, %bb16 ], [ %v111, %bb18 ]
  %v113 = call float @f16_to_f32(i16 %v112) #0
  br label %bb50
bb20:
  %v114 = add i64 %v45, 7
  %v115 = udiv i64 %v114, 8
  %v116 = mul i64 %v90, %v115
  %v117 = udiv i64 %v42, 8
  %v118 = add i64 %v116, %v117
  %v119 = urem i64 %v42, 8
  %v120 = mul i64 %v119, 4
  br label %bb22
bb21:
  %v121 = udiv i64 %v46, %v47
  %v122 = mul i64 %v42, %v121
  %v123 = add i64 %v122, %v90
  %v124 = udiv i64 %v123, 8
  %v125 = urem i64 %v123, 8
  %v126 = mul i64 %v125, 4
  br label %bb22
bb22:
  %v127 = phi i64 [ %v118, %bb20 ], [ %v124, %bb21 ]
  %v128 = phi i64 [ %v120, %bb20 ], [ %v126, %bb21 ]
  %v129 = extractvalue { ptr, i64 } %v28, 1
  %v130 = icmp ult i64 %v127, %v129
  br i1 %v130, label %bb23, label %bb65
bb23:
  %v131 = extractvalue { ptr, i64 } %v28, 0
  %v132 = getelementptr inbounds i32, ptr %v131, i64 %v127
  %v133 = load i32, ptr %v132, align 4
  %v134 = trunc i64 %v128 to i32
  %v135 = and i32 %v134, 31
  %v136 = lshr i32 %v133, %v135
  %v137 = and i32 %v136, 15
  %v138 = trunc i32 %v137 to i8
  %v139 = insertvalue { i64, i64 } undef, i64 0, 0
  %v140 = insertvalue { i64, i64 } %v139, i64 %v47, 1
  %v141 = extractvalue { i64, i64 } %v140, 0
  %v142 = extractvalue { i64, i64 } %v140, 1
  %v143 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v141, i64 %v142, i64 8) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v143, ptr %v36, align 8
  br label %bb51
bb24:
  %v144 = phi float [ %v144, %bb28 ], [ %v182, %bb41 ], [ %v73, %bb51 ]
  %v145 = phi i64 [ %v255, %bb28 ], [ %v255, %bb41 ], [ %v241, %bb51 ]
  %v146 = phi i64 [ %v256, %bb28 ], [ %v256, %bb41 ], [ %v244, %bb51 ]
  %v147 = add i64 %v246, 1
  %v148 = icmp eq i64 %v147, 0
  %v149 = select i1 %v148, i8 0, i8 1
  %v150 = insertvalue { i8, { { i64 } } } undef, i8 %v149, 0
  %v151 = insertvalue { i8, { { i64 } } } %v150, i64 %v147, 1, 0, 0
  %v152 = extractvalue { i8, { { i64 } } } %v151, 0
  %v153 = zext i8 %v152 to i64
  %v154 = icmp eq i64 %v153, 1
  %v155 = extractvalue { i8, { { i64 } } } %v151, 1
  %v156 = alloca { { i64 } }, align 8
  store { { i64 } } %v155, ptr %v156, align 8
  %v157 = load i64, ptr %v156, align 8
  %v158 = icmp ugt i64 %v146, 0
  %v159 = xor i1 %v158, 1
  br i1 %v159, label %bb53, label %bb52
bb25:
  %v160 = extractvalue { i64, i64 } %v254, 1
  %v161 = add i64 %v89, %v160
  %v162 = icmp uge i64 %v161, %v58
  %v163 = xor i1 %v162, 1
  br i1 %v163, label %bb27, label %bb26
bb26:
  br label %bb42
bb27:
  %v164 = add i64 %v161, 7
  %v165 = icmp ult i64 %v164, %v56
  %v166 = xor i1 %v165, 1
  br i1 %v166, label %bb29, label %bb28
bb28:
  br label %bb24
bb29:
  %v167 = xor i1 %v91, 1
  br i1 %v167, label %bb31, label %bb30
bb30:
  %v168 = zext i32 3 to i64
  %v169 = and i64 %v168, 63
  %v170 = lshr i64 %v161, %v169
  %v171 = mul i64 %v170, %v45
  %v172 = add i64 %v171, %v42
  br label %bb32
bb31:
  %v173 = mul i64 %v42, %v46
  %v174 = add i64 %v173, %v161
  %v175 = udiv i64 %v174, 8
  br label %bb32
bb32:
  %v176 = phi i64 [ %v172, %bb30 ], [ %v175, %bb31 ]
  %v177 = extractvalue { ptr, i64 } %v26, 1
  %v178 = icmp ult i64 %v176, %v177
  br i1 %v178, label %bb33, label %bb66
bb33:
  %v179 = extractvalue { ptr, i64 } %v26, 0
  %v180 = getelementptr inbounds i32, ptr %v179, i64 %v176
  %v181 = load i32, ptr %v180, align 4
  br label %bb34
bb34:
  %v182 = phi float [ %v144, %bb33 ], [ %v182, %bb38 ], [ %v213, %bb40 ]
  %v183 = phi i64 [ 0, %bb33 ], [ %v266, %bb38 ], [ %v266, %bb40 ]
  %v184 = icmp ult i64 %v183, 8
  %v185 = xor i1 %v184, 1
  br i1 %v185, label %bb57, label %bb56
bb35:
  %v186 = extractvalue { i64, i64 } %v265, 1
  %v187 = add i64 %v161, %v186
  %v188 = icmp uge i64 %v187, %v58
  %v189 = xor i1 %v188, 1
  br i1 %v189, label %bb37, label %bb36
bb36:
  br label %bb41
bb37:
  %v190 = icmp ult i64 %v187, %v56
  %v191 = xor i1 %v190, 1
  br i1 %v191, label %bb39, label %bb38
bb38:
  br label %bb34
bb39:
  %v192 = mul i64 %v186, 4
  %v193 = trunc i64 %v192 to i32
  %v194 = and i32 %v193, 31
  %v195 = lshr i32 %v181, %v194
  %v196 = and i32 %v195, 15
  %v197 = trunc i32 %v196 to i8
  %v198 = sitofp i8 %v197 to float
  %v199 = sitofp i8 %v138 to float
  %v200 = fadd contract float %v199, 1.0
  %v201 = fsub contract float %v198, %v200
  %v202 = fmul contract float %v201, %v113
  %v203 = extractvalue { ptr, i64 } %v29, 1
  %v204 = icmp ult i64 %v187, %v203
  br i1 %v204, label %bb40, label %bb67
bb40:
  %v205 = extractvalue { ptr, i64 } %v29, 0
  %v206 = getelementptr inbounds i16, ptr %v205, i64 %v187
  %v207 = load i16, ptr %v206, align 2
  %v208 = zext i16 %v207 to i32
  %v209 = and i32 16, 31
  %v210 = shl i32 %v208, %v209
  %v211 = bitcast i32 %v210 to float
  %v212 = fmul contract float %v202, %v211
  %v213 = fadd contract float %v182, %v212
  br label %bb34
bb41:
  br label %bb24
bb42:
  br label %bb11
bb43:
  %v214 = extractvalue { ptr, i64 } %v25, 0
  %v215 = getelementptr inbounds float, ptr %v214, i64 %v94
  store float %v73, ptr %v215, align 4
  br label %bb44
bb44:
  ret void
bb45:
  %v216 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v217 = getelementptr inbounds { i64, i64 }, ptr %v216, i32 0, i32 0
  %v218 = load i64, ptr %v217, align 8
  %v219 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 0
  %v220 = getelementptr inbounds { i64, i64 }, ptr %v219, i32 0, i32 1
  %v221 = load i64, ptr %v220, align 8
  %v222 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 1
  %v223 = load i64, ptr %v222, align 8
  %v224 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v35, i32 0, i32 2
  %v225 = load i1, ptr %v224, align 1
  br label %bb11
bb46:
  %v226 = add i64 %v74, %v86
  %v227 = sub i64 %v75, 1
  %v228 = insertvalue { i64, i64 } undef, i64 1, 0
  %v229 = insertvalue { i64, i64 } %v228, i64 %v74, 1
  br label %bb48
bb47:
  %v230 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb48
bb48:
  %v231 = phi { i64, i64 } [ %v229, %bb46 ], [ %v230, %bb47 ]
  %v232 = phi i64 [ %v226, %bb46 ], [ %v74, %bb47 ]
  %v233 = phi i64 [ %v227, %bb46 ], [ %v75, %bb47 ]
  %v234 = extractvalue { i64, i64 } %v231, 0
  %v235 = bitcast i64 %v234 to i64
  %v236 = icmp eq i64 %v235, 0
  br i1 %v236, label %bb14, label %bb49
bb49:
  %v237 = icmp eq i64 %v235, 1
  br i1 %v237, label %bb13, label %bb12
bb50:
  %v238 = xor i1 %v91, 1
  br i1 %v238, label %bb21, label %bb20
bb51:
  %v239 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 0
  %v240 = getelementptr inbounds { i64, i64 }, ptr %v239, i32 0, i32 0
  %v241 = load i64, ptr %v240, align 8
  %v242 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 0
  %v243 = getelementptr inbounds { i64, i64 }, ptr %v242, i32 0, i32 1
  %v244 = load i64, ptr %v243, align 8
  %v245 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 1
  %v246 = load i64, ptr %v245, align 8
  %v247 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v36, i32 0, i32 2
  %v248 = load i1, ptr %v247, align 1
  br label %bb24
bb52:
  %v249 = add i64 %v145, %v157
  %v250 = sub i64 %v146, 1
  %v251 = insertvalue { i64, i64 } undef, i64 1, 0
  %v252 = insertvalue { i64, i64 } %v251, i64 %v145, 1
  br label %bb54
bb53:
  %v253 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb54
bb54:
  %v254 = phi { i64, i64 } [ %v252, %bb52 ], [ %v253, %bb53 ]
  %v255 = phi i64 [ %v249, %bb52 ], [ %v145, %bb53 ]
  %v256 = phi i64 [ %v250, %bb52 ], [ %v146, %bb53 ]
  %v257 = extractvalue { i64, i64 } %v254, 0
  %v258 = bitcast i64 %v257 to i64
  %v259 = icmp eq i64 %v258, 0
  br i1 %v259, label %bb42, label %bb55
bb55:
  %v260 = icmp eq i64 %v258, 1
  br i1 %v260, label %bb25, label %bb12
bb56:
  %v261 = add i64 %v183, 1
  %v262 = insertvalue { i64, i64 } undef, i64 1, 0
  %v263 = insertvalue { i64, i64 } %v262, i64 %v183, 1
  br label %bb58
bb57:
  %v264 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb58
bb58:
  %v265 = phi { i64, i64 } [ %v263, %bb56 ], [ %v264, %bb57 ]
  %v266 = phi i64 [ %v261, %bb56 ], [ %v183, %bb57 ]
  %v267 = extractvalue { i64, i64 } %v265, 0
  %v268 = bitcast i64 %v267 to i64
  %v269 = icmp eq i64 %v268, 0
  br i1 %v269, label %bb41, label %bb59
bb59:
  %v270 = icmp eq i64 %v268, 1
  br i1 %v270, label %bb35, label %bb12
bb60:
  unreachable
bb61:
  unreachable
bb62:
  unreachable
bb63:
  unreachable
bb64:
  unreachable
bb65:
  unreachable
bb66:
  unreachable
bb67:
  unreachable
}

declare float @__nv_sqrtf(float)
declare float @__nv_expf(float)
declare float @__nv_logf(float)

define void @infers_gdn_chunked_gated_delta_prefill_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, ptr %v12, i64 %v13, ptr %v14, i64 %v15, ptr %v16, i64 %v17, i32 %v18, i32 %v19, i32 %v20, i32 %v21, i32 %v22) #0 {
entry:
  %v23 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v1, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v3, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v5, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v7, 1
  %v31 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v32 = insertvalue { ptr, i64 } %v31, i64 %v9, 1
  %v33 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v34 = insertvalue { ptr, i64 } %v33, i64 %v11, 1
  %v35 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v36 = insertvalue { ptr, i64 } %v35, i64 %v13, 1
  %v37 = insertvalue { ptr, i64 } undef, ptr %v14, 0
  %v38 = insertvalue { ptr, i64 } %v37, i64 %v15, 1
  %v39 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v40 = insertvalue { ptr, i64 } %v39, i64 %v17, 1
  br label %bb0
bb0:
  %v41 = phi { ptr, i64 } [ %v24, %entry ]
  %v42 = phi { ptr, i64 } [ %v26, %entry ]
  %v43 = phi { ptr, i64 } [ %v28, %entry ]
  %v44 = phi { ptr, i64 } [ %v30, %entry ]
  %v45 = phi { ptr, i64 } [ %v32, %entry ]
  %v46 = phi { ptr, i64 } [ %v34, %entry ]
  %v47 = phi { ptr, i64 } [ %v36, %entry ]
  %v48 = phi { ptr, i64 } [ %v38, %entry ]
  %v49 = phi { ptr, i64 } [ %v40, %entry ]
  %v50 = phi i32 [ %v18, %entry ]
  %v51 = phi i32 [ %v19, %entry ]
  %v52 = phi i32 [ %v20, %entry ]
  %v53 = phi i32 [ %v21, %entry ]
  %v54 = phi i32 [ %v22, %entry ]
  %v55 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v56 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v57 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v58 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v59 = alloca [128 x float], align 4
  %v60 = alloca { i64, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v62 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v63 = zext i32 %v62 to i64
  %v64 = zext i32 %v54 to i64
  %v65 = zext i32 %v52 to i64
  %v66 = zext i32 %v53 to i64
  %v67 = zext i32 %v50 to i64
  %v68 = add i64 %v67, %v64
  %v69 = sub i64 %v68, 1
  %v70 = icmp eq i64 %v64, 0
  %v71 = xor i1 %v70, 1
  br i1 %v71, label %bb3, label %bb306
bb3:
  %v72 = udiv i64 %v69, %v64
  %v73 = uitofp i64 %v65 to float
  %v74 = call float @__nv_sqrtf(float %v73) #0
  br label %bb189
bb4:
  %v75 = extractvalue { ptr, i64 } %v46, 0
  %v76 = getelementptr inbounds float, ptr %v75, i64 %v63
  %v77 = load float, ptr %v76, align 4
  %v78 = call float @__nv_expf(float %v77) #0
  br label %bb5
bb5:
  br label %bb6
bb6:
  %v79 = mul i64 %v64, %v65
  %v80 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v79
  %v81 = addrspacecast ptr addrspace(3) %v80 to ptr
  %v82 = mul i64 2, %v64
  %v83 = mul i64 %v82, %v65
  %v84 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v83
  %v85 = addrspacecast ptr addrspace(3) %v84 to ptr
  %v86 = mul i64 %v64, %v64
  %v87 = add i64 %v83, %v86
  %v88 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v87
  %v89 = addrspacecast ptr addrspace(3) %v88 to ptr
  %v90 = getelementptr inbounds float, ptr %v89, i64 %v64
  %v91 = getelementptr inbounds float, ptr %v90, i64 %v64
  %v92 = mul i64 %v63, %v65
  %v93 = mul i64 %v92, %v66
  br label %bb7
bb7:
  %v94 = phi i64 [ 0, %bb6 ], [ %v880, %bb188 ]
  %v95 = icmp ult i64 %v94, %v72
  %v96 = xor i1 %v95, 1
  br i1 %v96, label %bb191, label %bb190
bb8:
  unreachable
bb9:
  %v97 = extractvalue { i64, i64 } %v879, 1
  %v98 = mul i64 %v97, %v64
  %v99 = sub i64 %v67, %v98
  %v100 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v64, i64 %v99) #0
  br label %bb11
bb10:
  ret void
bb11:
  %v101 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb12
bb12:
  %v102 = zext i32 %v101 to i64
  %v103 = insertvalue { i64, i64 } undef, i64 0, 0
  %v104 = insertvalue { i64, i64 } %v103, i64 %v64, 1
  %v105 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb13
bb13:
  %v106 = zext i32 %v105 to i64
  %v107 = extractvalue { i64, i64 } %v104, 0
  %v108 = extractvalue { i64, i64 } %v104, 1
  %v109 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v107, i64 %v108, i64 %v106) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v109, ptr %v55, align 8
  br label %bb194
bb14:
  %v110 = phi i64 [ %v901, %bb17 ], [ %v901, %bb30 ], [ %v887, %bb194 ]
  %v111 = phi i64 [ %v902, %bb17 ], [ %v902, %bb30 ], [ %v890, %bb194 ]
  %v112 = add i64 %v892, 1
  %v113 = icmp eq i64 %v112, 0
  %v114 = select i1 %v113, i8 0, i8 1
  %v115 = insertvalue { i8, { { i64 } } } undef, i8 %v114, 0
  %v116 = insertvalue { i8, { { i64 } } } %v115, i64 %v112, 1, 0, 0
  %v117 = extractvalue { i8, { { i64 } } } %v116, 0
  %v118 = zext i8 %v117 to i64
  %v119 = icmp eq i64 %v118, 1
  %v120 = extractvalue { i8, { { i64 } } } %v116, 1
  %v121 = alloca { { i64 } }, align 8
  store { { i64 } } %v120, ptr %v121, align 8
  %v122 = load i64, ptr %v121, align 8
  %v123 = icmp ugt i64 %v111, 0
  %v124 = xor i1 %v123, 1
  br i1 %v124, label %bb196, label %bb195
bb15:
  %v125 = extractvalue { i64, i64 } %v900, 1
  %v126 = add i64 %v102, %v125
  %v127 = icmp uge i64 %v126, %v100
  %v128 = xor i1 %v127, 1
  br i1 %v128, label %bb18, label %bb17
bb16:
  call void @llvm.nvvm.barrier0() #0
  br label %bb31
bb17:
  br label %bb14
bb18:
  %v130 = add i64 %v98, %v126
  %v131 = zext i32 %v51 to i64
  %v132 = mul i64 %v130, %v131
  %v133 = add i64 %v132, %v63
  %v134 = extractvalue { ptr, i64 } %v44, 1
  %v135 = icmp ult i64 %v133, %v134
  br i1 %v135, label %bb19, label %bb307
bb19:
  %v136 = extractvalue { ptr, i64 } %v44, 0
  %v137 = getelementptr inbounds i16, ptr %v136, i64 %v133
  %v138 = load i16, ptr %v137, align 2
  %v139 = zext i16 %v138 to i32
  %v140 = and i32 16, 31
  %v141 = shl i32 %v139, %v140
  %v142 = bitcast i32 %v141 to float
  %v143 = extractvalue { ptr, i64 } %v47, 1
  %v144 = icmp ult i64 %v63, %v143
  br i1 %v144, label %bb20, label %bb308
bb20:
  %v145 = extractvalue { ptr, i64 } %v47, 0
  %v146 = getelementptr inbounds float, ptr %v145, i64 %v63
  %v147 = load float, ptr %v146, align 4
  %v148 = fadd contract float %v142, %v147
  %v149 = fcmp ogt float %v148, 20.0
  %v150 = xor i1 %v149, 1
  br i1 %v150, label %bb22, label %bb21
bb21:
  br label %bb28
bb22:
  %v151 = fcmp olt float %v148, -20.0
  %v152 = xor i1 %v151, 1
  br i1 %v152, label %bb24, label %bb23
bb23:
  br label %bb27
bb24:
  %v153 = call float @__nv_expf(float %v148) #0
  br label %bb25
bb25:
  %v154 = fadd contract float 1.0, %v153
  %v155 = call float @__nv_logf(float %v154) #0
  br label %bb26
bb26:
  br label %bb27
bb27:
  %v156 = phi float [ 0.0, %bb23 ], [ %v155, %bb26 ]
  br label %bb28
bb28:
  %v157 = phi float [ %v148, %bb21 ], [ %v156, %bb27 ]
  %v158 = fneg float %v78
  %v159 = fmul contract float %v158, %v157
  %v160 = extractvalue { ptr, i64 } %v45, 1
  %v161 = icmp ult i64 %v133, %v160
  br i1 %v161, label %bb29, label %bb309
bb29:
  %v162 = extractvalue { ptr, i64 } %v45, 0
  %v163 = getelementptr inbounds i16, ptr %v162, i64 %v133
  %v164 = load i16, ptr %v163, align 2
  %v165 = zext i16 %v164 to i32
  %v166 = and i32 16, 31
  %v167 = shl i32 %v165, %v166
  %v168 = bitcast i32 %v167 to float
  %v169 = fneg float %v168
  %v170 = call float @__nv_expf(float %v169) #0
  br label %bb30
bb30:
  %v171 = fadd contract float 1.0, %v170
  %v172 = fdiv contract float 1.0, %v171
  %v173 = getelementptr inbounds float, ptr %v89, i64 %v126
  store float %v159, ptr %v173, align 4
  %v174 = getelementptr inbounds float, ptr %v90, i64 %v126
  store float %v172, ptr %v174, align 4
  br label %bb14
bb31:
  %v175 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb32
bb32:
  %v176 = zext i32 %v175 to i64
  %v177 = insertvalue { i64, i64 } undef, i64 %v176, 0
  %v178 = insertvalue { i64, i64 } %v177, i64 %v64, 1
  %v179 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb33
bb33:
  %v180 = zext i32 %v179 to i64
  %v181 = extractvalue { i64, i64 } %v178, 0
  %v182 = extractvalue { i64, i64 } %v178, 1
  %v183 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v181, i64 %v182, i64 %v180) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v183, ptr %v56, align 8
  br label %bb199
bb34:
  %v184 = phi i64 [ %v923, %bb38 ], [ %v909, %bb199 ]
  %v185 = phi i64 [ %v924, %bb38 ], [ %v912, %bb199 ]
  %v186 = add i64 %v914, 1
  %v187 = icmp eq i64 %v186, 0
  %v188 = select i1 %v187, i8 0, i8 1
  %v189 = insertvalue { i8, { { i64 } } } undef, i8 %v188, 0
  %v190 = insertvalue { i8, { { i64 } } } %v189, i64 %v186, 1, 0, 0
  %v191 = extractvalue { i8, { { i64 } } } %v190, 0
  %v192 = zext i8 %v191 to i64
  %v193 = icmp eq i64 %v192, 1
  %v194 = extractvalue { i8, { { i64 } } } %v190, 1
  %v195 = alloca { { i64 } }, align 8
  store { { i64 } } %v194, ptr %v195, align 8
  %v196 = load i64, ptr %v195, align 8
  %v197 = icmp ugt i64 %v185, 0
  %v198 = xor i1 %v197, 1
  br i1 %v198, label %bb201, label %bb200
bb35:
  %v199 = extractvalue { i64, i64 } %v922, 1
  %v200 = icmp uge i64 %v199, %v100
  %v201 = xor i1 %v200, 1
  br i1 %v201, label %bb38, label %bb37
bb36:
  call void @llvm.nvvm.barrier0() #0
  br label %bb39
bb37:
  %v203 = getelementptr inbounds float, ptr %v89, i64 %v199
  store float 0.0, ptr %v203, align 4
  %v204 = getelementptr inbounds float, ptr %v90, i64 %v199
  store float 0.0, ptr %v204, align 4
  br label %bb38
bb38:
  br label %bb34
bb39:
  %v205 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb40
bb40:
  %v206 = icmp eq i32 %v205, 0
  br i1 %v206, label %bb41, label %bb45
bb41:
  br label %bb42
bb42:
  %v207 = phi float [ 0.0, %bb41 ], [ %v214, %bb43 ]
  %v208 = phi i64 [ 0, %bb41 ], [ %v934, %bb43 ]
  %v209 = icmp ult i64 %v208, %v64
  %v210 = xor i1 %v209, 1
  br i1 %v210, label %bb205, label %bb204
bb43:
  %v211 = extractvalue { i64, i64 } %v933, 1
  %v212 = getelementptr inbounds float, ptr %v89, i64 %v211
  %v213 = load float, ptr %v212, align 4
  %v214 = fadd contract float %v207, %v213
  store float %v214, ptr %v212, align 4
  br label %bb42
bb44:
  br label %bb46
bb45:
  br label %bb46
bb46:
  call void @llvm.nvvm.barrier0() #0
  br label %bb47
bb47:
  %v216 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb48
bb48:
  %v217 = zext i32 %v216 to i64
  %v218 = insertvalue { i64, i64 } undef, i64 0, 0
  %v219 = insertvalue { i64, i64 } %v218, i64 %v79, 1
  %v220 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb49
bb49:
  %v221 = zext i32 %v220 to i64
  %v222 = extractvalue { i64, i64 } %v219, 0
  %v223 = extractvalue { i64, i64 } %v219, 1
  %v224 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v222, i64 %v223, i64 %v221) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v224, ptr %v57, align 8
  br label %bb208
bb50:
  %v225 = phi i64 [ %v955, %bb58 ], [ %v955, %bb59 ], [ %v941, %bb208 ]
  %v226 = phi i64 [ %v956, %bb58 ], [ %v956, %bb59 ], [ %v944, %bb208 ]
  %v227 = add i64 %v946, 1
  %v228 = icmp eq i64 %v227, 0
  %v229 = select i1 %v228, i8 0, i8 1
  %v230 = insertvalue { i8, { { i64 } } } undef, i8 %v229, 0
  %v231 = insertvalue { i8, { { i64 } } } %v230, i64 %v227, 1, 0, 0
  %v232 = extractvalue { i8, { { i64 } } } %v231, 0
  %v233 = zext i8 %v232 to i64
  %v234 = icmp eq i64 %v233, 1
  %v235 = extractvalue { i8, { { i64 } } } %v231, 1
  %v236 = alloca { { i64 } }, align 8
  store { { i64 } } %v235, ptr %v236, align 8
  %v237 = load i64, ptr %v236, align 8
  %v238 = icmp ugt i64 %v226, 0
  %v239 = xor i1 %v238, 1
  br i1 %v239, label %bb210, label %bb209
bb51:
  %v240 = extractvalue { i64, i64 } %v954, 1
  %v241 = add i64 %v217, %v240
  %v242 = icmp uge i64 %v241, %v79
  %v243 = xor i1 %v242, 1
  br i1 %v243, label %bb54, label %bb53
bb52:
  %v244 = icmp ult i64 %v217, %v100
  %v245 = xor i1 %v244, 1
  br i1 %v245, label %bb67, label %bb60
bb53:
  br label %bb59
bb54:
  %v246 = icmp eq i64 %v65, 0
  %v247 = xor i1 %v246, 1
  br i1 %v247, label %bb55, label %bb310
bb55:
  %v248 = udiv i64 %v241, %v65
  %v249 = urem i64 %v241, %v65
  %v250 = icmp uge i64 %v248, %v100
  %v251 = xor i1 %v250, 1
  br i1 %v251, label %bb57, label %bb56
bb56:
  %v252 = mul i64 %v248, %v65
  %v253 = add i64 %v252, %v249
  %v254 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v253
  %v255 = addrspacecast ptr addrspace(3) %v254 to ptr
  store float 0.0, ptr %v255, align 4
  br label %bb59
bb57:
  %v256 = add i64 %v98, %v248
  %v257 = zext i32 %v51 to i64
  %v258 = mul i64 %v256, %v257
  %v259 = mul i64 %v258, %v65
  %v260 = add i64 %v259, %v92
  %v261 = add i64 %v260, %v249
  %v262 = extractvalue { ptr, i64 } %v42, 1
  %v263 = icmp ult i64 %v261, %v262
  br i1 %v263, label %bb58, label %bb311
bb58:
  %v264 = extractvalue { ptr, i64 } %v42, 0
  %v265 = getelementptr inbounds i16, ptr %v264, i64 %v261
  %v266 = load i16, ptr %v265, align 2
  %v267 = zext i16 %v266 to i32
  %v268 = and i32 16, 31
  %v269 = shl i32 %v267, %v268
  %v270 = bitcast i32 %v269 to float
  %v271 = mul i64 %v248, %v65
  %v272 = add i64 %v271, %v249
  %v273 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v272
  %v274 = addrspacecast ptr addrspace(3) %v273 to ptr
  store float %v270, ptr %v274, align 4
  br label %bb50
bb59:
  br label %bb50
bb60:
  br label %bb61
bb61:
  %v275 = phi float [ 0.0, %bb60 ], [ %v286, %bb62 ]
  %v276 = phi i64 [ 0, %bb60 ], [ %v966, %bb62 ]
  %v277 = icmp ult i64 %v276, %v65
  %v278 = xor i1 %v277, 1
  br i1 %v278, label %bb214, label %bb213
bb62:
  %v279 = extractvalue { i64, i64 } %v965, 1
  %v280 = mul i64 %v217, %v65
  %v281 = add i64 %v280, %v279
  %v282 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v281
  %v283 = addrspacecast ptr addrspace(3) %v282 to ptr
  %v284 = load float, ptr %v283, align 4
  %v285 = fmul contract float %v284, %v284
  %v286 = fadd contract float %v275, %v285
  br label %bb61
bb63:
  %v287 = fadd contract float %v275, 0.0000009999999974752427
  %v288 = call float @__nv_sqrtf(float %v287) #0
  br label %bb217
bb64:
  %v289 = phi i64 [ %v977, %bb65 ], [ 0, %bb217 ]
  %v290 = icmp ult i64 %v289, %v65
  %v291 = xor i1 %v290, 1
  br i1 %v291, label %bb219, label %bb218
bb65:
  %v292 = extractvalue { i64, i64 } %v976, 1
  %v293 = mul i64 %v217, %v65
  %v294 = add i64 %v293, %v292
  %v295 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v294
  %v296 = addrspacecast ptr addrspace(3) %v295 to ptr
  %v297 = load float, ptr %v296, align 4
  %v298 = fmul contract float %v297, %v971
  store float %v298, ptr %v296, align 4
  br label %bb64
bb66:
  br label %bb67
bb67:
  call void @llvm.nvvm.barrier0() #0
  br label %bb68
bb68:
  %v300 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb69
bb69:
  %v301 = zext i32 %v300 to i64
  %v302 = extractvalue { i64, i64 } %v219, 0
  %v303 = extractvalue { i64, i64 } %v219, 1
  %v304 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v302, i64 %v303, i64 %v301) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v304, ptr %v58, align 8
  br label %bb222
bb70:
  %v305 = phi i64 [ %v998, %bb73 ], [ %v998, %bb78 ], [ %v984, %bb222 ]
  %v306 = phi i64 [ %v999, %bb73 ], [ %v999, %bb78 ], [ %v987, %bb222 ]
  %v307 = add i64 %v989, 1
  %v308 = icmp eq i64 %v307, 0
  %v309 = select i1 %v308, i8 0, i8 1
  %v310 = insertvalue { i8, { { i64 } } } undef, i8 %v309, 0
  %v311 = insertvalue { i8, { { i64 } } } %v310, i64 %v307, 1, 0, 0
  %v312 = extractvalue { i8, { { i64 } } } %v311, 0
  %v313 = zext i8 %v312 to i64
  %v314 = icmp eq i64 %v313, 1
  %v315 = extractvalue { i8, { { i64 } } } %v311, 1
  %v316 = alloca { { i64 } }, align 8
  store { { i64 } } %v315, ptr %v316, align 8
  %v317 = load i64, ptr %v316, align 8
  %v318 = icmp ugt i64 %v306, 0
  %v319 = xor i1 %v318, 1
  br i1 %v319, label %bb224, label %bb223
bb71:
  %v320 = extractvalue { i64, i64 } %v997, 1
  %v321 = add i64 %v217, %v320
  %v322 = icmp uge i64 %v321, %v79
  %v323 = xor i1 %v322, 1
  br i1 %v323, label %bb74, label %bb73
bb72:
  call void @llvm.nvvm.barrier0() #0
  br label %bb79
bb73:
  br label %bb70
bb74:
  %v325 = icmp eq i64 %v65, 0
  %v326 = xor i1 %v325, 1
  br i1 %v326, label %bb75, label %bb312
bb75:
  %v327 = udiv i64 %v321, %v65
  %v328 = urem i64 %v321, %v65
  %v329 = mul i64 %v327, %v65
  %v330 = add i64 %v329, %v328
  %v331 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v330
  %v332 = addrspacecast ptr addrspace(3) %v331 to ptr
  %v333 = load float, ptr %v332, align 4
  %v334 = icmp ult i64 %v327, %v100
  %v335 = xor i1 %v334, 1
  br i1 %v335, label %bb77, label %bb76
bb76:
  %v336 = getelementptr inbounds float, ptr %v90, i64 %v327
  %v337 = load float, ptr %v336, align 4
  br label %bb78
bb77:
  br label %bb78
bb78:
  %v338 = phi float [ %v337, %bb76 ], [ 0.0, %bb77 ]
  %v339 = getelementptr inbounds float, ptr %v81, i64 %v330
  %v340 = fmul contract float %v333, %v338
  store float %v340, ptr %v339, align 4
  br label %bb70
bb79:
  %v341 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb80
bb80:
  %v342 = zext i32 %v341 to i64
  %v343 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb81
bb81:
  %v344 = zext i32 %v343 to i64
  %v345 = add i64 %v86, %v344
  %v346 = sub i64 %v345, 1
  %v347 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb82
bb82:
  %v348 = zext i32 %v347 to i64
  %v349 = icmp eq i64 %v348, 0
  %v350 = xor i1 %v349, 1
  br i1 %v350, label %bb83, label %bb313
bb83:
  %v351 = udiv i64 %v346, %v348
  %v352 = mul i64 %v342, %v351
  %v353 = add i64 %v342, 1
  %v354 = mul i64 %v353, %v351
  br label %bb84
bb84:
  %v355 = phi i64 [ %v352, %bb83 ], [ %v1009, %bb94 ]
  %v356 = icmp ult i64 %v355, %v354
  %v357 = xor i1 %v356, 1
  br i1 %v357, label %bb228, label %bb227
bb85:
  %v358 = extractvalue { i64, i64 } %v1008, 1
  %v359 = icmp uge i64 %v358, %v86
  %v360 = xor i1 %v359, 1
  br i1 %v360, label %bb87, label %bb86
bb86:
  br label %bb95
bb87:
  %v361 = udiv i64 %v358, %v64
  %v362 = urem i64 %v358, %v64
  br label %bb88
bb88:
  %v363 = phi float [ 0.0, %bb87 ], [ %v378, %bb89 ]
  %v364 = phi i64 [ 0, %bb87 ], [ %v1019, %bb89 ]
  %v365 = icmp ult i64 %v364, %v65
  %v366 = xor i1 %v365, 1
  br i1 %v366, label %bb232, label %bb231
bb89:
  %v367 = extractvalue { i64, i64 } %v1018, 1
  %v368 = mul i64 %v361, %v65
  %v369 = add i64 %v368, %v367
  %v370 = getelementptr inbounds float, ptr %v81, i64 %v369
  %v371 = load float, ptr %v370, align 4
  %v372 = mul i64 %v362, %v65
  %v373 = add i64 %v372, %v367
  %v374 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v373
  %v375 = addrspacecast ptr addrspace(3) %v374 to ptr
  %v376 = load float, ptr %v375, align 4
  %v377 = fmul contract float %v371, %v376
  %v378 = fadd contract float %v363, %v377
  br label %bb88
bb90:
  %v379 = icmp ugt i64 %v361, %v362
  %v380 = xor i1 %v379, 1
  br i1 %v380, label %bb92, label %bb91
bb91:
  %v381 = getelementptr inbounds float, ptr %v89, i64 %v361
  %v382 = load float, ptr %v381, align 4
  %v383 = getelementptr inbounds float, ptr %v89, i64 %v362
  %v384 = load float, ptr %v383, align 4
  %v385 = fsub contract float %v382, %v384
  %v386 = fneg float %v363
  %v387 = call float @__nv_expf(float %v385) #0
  br label %bb93
bb92:
  br label %bb94
bb93:
  %v388 = fmul contract float %v386, %v387
  br label %bb94
bb94:
  %v389 = phi float [ 0.0, %bb92 ], [ %v388, %bb93 ]
  %v390 = mul i64 %v361, %v64
  %v391 = add i64 %v390, %v362
  %v392 = getelementptr inbounds float, ptr %v85, i64 %v391
  store float %v389, ptr %v392, align 4
  br label %bb84
bb95:
  call void @llvm.nvvm.barrier0() #0
  br label %bb96
bb96:
  %v394 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb97
bb97:
  %v395 = icmp eq i32 %v394, 0
  br i1 %v395, label %bb98, label %bb114
bb98:
  br label %bb99
bb99:
  %v396 = phi i64 [ 1, %bb98 ], [ %v1029, %bb107 ]
  %v397 = icmp ult i64 %v396, %v100
  %v398 = xor i1 %v397, 1
  br i1 %v398, label %bb236, label %bb235
bb100:
  %v399 = extractvalue { i64, i64 } %v1028, 1
  br label %bb102
bb101:
  br label %bb111
bb102:
  %v400 = phi i64 [ 0, %bb100 ], [ %v1039, %bb103 ]
  %v401 = icmp ult i64 %v400, %v399
  %v402 = xor i1 %v401, 1
  br i1 %v402, label %bb240, label %bb239
bb103:
  %v403 = extractvalue { i64, i64 } %v1038, 1
  %v404 = mul i64 %v399, %v64
  %v405 = add i64 %v404, %v403
  %v406 = getelementptr inbounds float, ptr %v85, i64 %v405
  %v407 = load float, ptr %v406, align 4
  %v408 = getelementptr inbounds float, ptr %v91, i64 %v403
  store float %v407, ptr %v408, align 4
  br label %bb102
bb104:
  br label %bb105
bb105:
  %v409 = phi i64 [ 0, %bb104 ], [ %v1049, %bb110 ]
  %v410 = icmp ult i64 %v409, %v399
  %v411 = xor i1 %v410, 1
  br i1 %v411, label %bb244, label %bb243
bb106:
  %v412 = extractvalue { i64, i64 } %v1048, 1
  br label %bb108
bb107:
  br label %bb99
bb108:
  %v413 = phi float [ 0.0, %bb106 ], [ %v425, %bb109 ]
  %v414 = phi i64 [ 0, %bb106 ], [ %v1059, %bb109 ]
  %v415 = icmp ult i64 %v414, %v399
  %v416 = xor i1 %v415, 1
  br i1 %v416, label %bb248, label %bb247
bb109:
  %v417 = extractvalue { i64, i64 } %v1058, 1
  %v418 = getelementptr inbounds float, ptr %v91, i64 %v417
  %v419 = load float, ptr %v418, align 4
  %v420 = mul i64 %v417, %v64
  %v421 = add i64 %v420, %v412
  %v422 = getelementptr inbounds float, ptr %v85, i64 %v421
  %v423 = load float, ptr %v422, align 4
  %v424 = fmul contract float %v419, %v423
  %v425 = fadd contract float %v413, %v424
  br label %bb108
bb110:
  %v426 = getelementptr inbounds float, ptr %v91, i64 %v412
  %v427 = load float, ptr %v426, align 4
  %v428 = mul i64 %v399, %v64
  %v429 = add i64 %v428, %v412
  %v430 = getelementptr inbounds float, ptr %v85, i64 %v429
  %v431 = fadd contract float %v427, %v413
  store float %v431, ptr %v430, align 4
  br label %bb105
bb111:
  %v432 = phi i64 [ 0, %bb101 ], [ %v1069, %bb112 ]
  %v433 = icmp ult i64 %v432, %v100
  %v434 = xor i1 %v433, 1
  br i1 %v434, label %bb252, label %bb251
bb112:
  %v435 = extractvalue { i64, i64 } %v1068, 1
  %v436 = mul i64 %v435, %v64
  %v437 = add i64 %v436, %v435
  %v438 = getelementptr inbounds float, ptr %v85, i64 %v437
  %v439 = load float, ptr %v438, align 4
  %v440 = fadd contract float %v439, 1.0
  store float %v440, ptr %v438, align 4
  br label %bb111
bb113:
  br label %bb115
bb114:
  br label %bb115
bb115:
  call void @llvm.nvvm.barrier0() #0
  br label %bb116
bb116:
  %v442 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb117
bb117:
  %v443 = zext i32 %v442 to i64
  %v444 = mul i64 %v64, %v66
  %v445 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb118
bb118:
  %v446 = zext i32 %v445 to i64
  %v447 = add i64 %v444, %v446
  %v448 = sub i64 %v447, 1
  %v449 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb119
bb119:
  %v450 = zext i32 %v449 to i64
  %v451 = icmp eq i64 %v450, 0
  %v452 = xor i1 %v451, 1
  br i1 %v452, label %bb120, label %bb314
bb120:
  %v453 = udiv i64 %v448, %v450
  %v454 = mul i64 %v443, %v453
  %v455 = add i64 %v443, 1
  %v456 = mul i64 %v455, %v453
  br label %bb121
bb121:
  %v457 = phi i64 [ %v454, %bb120 ], [ %v1079, %bb141 ], [ %v1079, %bb160 ]
  %v458 = icmp ult i64 %v457, %v456
  %v459 = xor i1 %v458, 1
  br i1 %v459, label %bb256, label %bb255
bb122:
  %v460 = extractvalue { i64, i64 } %v1078, 1
  %v461 = icmp uge i64 %v460, %v444
  %v462 = xor i1 %v461, 1
  br i1 %v462, label %bb124, label %bb123
bb123:
  br label %bb159
bb124:
  %v463 = icmp eq i64 %v66, 0
  %v464 = xor i1 %v463, 1
  br i1 %v464, label %bb125, label %bb315
bb125:
  %v465 = udiv i64 %v460, %v66
  %v466 = urem i64 %v460, %v66
  %v467 = icmp uge i64 %v465, %v100
  %v468 = xor i1 %v467, 1
  br i1 %v468, label %bb126, label %bb160
bb126:
  %v469 = icmp uge i64 %v466, %v66
  %v470 = xor i1 %v469, 1
  br i1 %v470, label %bb127, label %bb160
bb127:
  %v471 = add i64 %v98, %v465
  %v472 = zext i32 %v51 to i64
  %v473 = mul i64 %v471, %v472
  %v474 = mul i64 %v473, %v65
  %v475 = add i64 %v474, %v92
  %v476 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 0
  store float 0.0, ptr %v476, align 4
  %v477 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 1
  store float 0.0, ptr %v477, align 4
  %v478 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 2
  store float 0.0, ptr %v478, align 4
  %v479 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 3
  store float 0.0, ptr %v479, align 4
  %v480 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 4
  store float 0.0, ptr %v480, align 4
  %v481 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 5
  store float 0.0, ptr %v481, align 4
  %v482 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 6
  store float 0.0, ptr %v482, align 4
  %v483 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 7
  store float 0.0, ptr %v483, align 4
  %v484 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 8
  store float 0.0, ptr %v484, align 4
  %v485 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 9
  store float 0.0, ptr %v485, align 4
  %v486 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 10
  store float 0.0, ptr %v486, align 4
  %v487 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 11
  store float 0.0, ptr %v487, align 4
  %v488 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 12
  store float 0.0, ptr %v488, align 4
  %v489 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 13
  store float 0.0, ptr %v489, align 4
  %v490 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 14
  store float 0.0, ptr %v490, align 4
  %v491 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 15
  store float 0.0, ptr %v491, align 4
  %v492 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 16
  store float 0.0, ptr %v492, align 4
  %v493 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 17
  store float 0.0, ptr %v493, align 4
  %v494 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 18
  store float 0.0, ptr %v494, align 4
  %v495 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 19
  store float 0.0, ptr %v495, align 4
  %v496 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 20
  store float 0.0, ptr %v496, align 4
  %v497 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 21
  store float 0.0, ptr %v497, align 4
  %v498 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 22
  store float 0.0, ptr %v498, align 4
  %v499 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 23
  store float 0.0, ptr %v499, align 4
  %v500 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 24
  store float 0.0, ptr %v500, align 4
  %v501 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 25
  store float 0.0, ptr %v501, align 4
  %v502 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 26
  store float 0.0, ptr %v502, align 4
  %v503 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 27
  store float 0.0, ptr %v503, align 4
  %v504 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 28
  store float 0.0, ptr %v504, align 4
  %v505 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 29
  store float 0.0, ptr %v505, align 4
  %v506 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 30
  store float 0.0, ptr %v506, align 4
  %v507 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 31
  store float 0.0, ptr %v507, align 4
  %v508 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 32
  store float 0.0, ptr %v508, align 4
  %v509 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 33
  store float 0.0, ptr %v509, align 4
  %v510 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 34
  store float 0.0, ptr %v510, align 4
  %v511 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 35
  store float 0.0, ptr %v511, align 4
  %v512 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 36
  store float 0.0, ptr %v512, align 4
  %v513 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 37
  store float 0.0, ptr %v513, align 4
  %v514 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 38
  store float 0.0, ptr %v514, align 4
  %v515 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 39
  store float 0.0, ptr %v515, align 4
  %v516 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 40
  store float 0.0, ptr %v516, align 4
  %v517 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 41
  store float 0.0, ptr %v517, align 4
  %v518 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 42
  store float 0.0, ptr %v518, align 4
  %v519 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 43
  store float 0.0, ptr %v519, align 4
  %v520 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 44
  store float 0.0, ptr %v520, align 4
  %v521 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 45
  store float 0.0, ptr %v521, align 4
  %v522 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 46
  store float 0.0, ptr %v522, align 4
  %v523 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 47
  store float 0.0, ptr %v523, align 4
  %v524 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 48
  store float 0.0, ptr %v524, align 4
  %v525 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 49
  store float 0.0, ptr %v525, align 4
  %v526 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 50
  store float 0.0, ptr %v526, align 4
  %v527 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 51
  store float 0.0, ptr %v527, align 4
  %v528 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 52
  store float 0.0, ptr %v528, align 4
  %v529 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 53
  store float 0.0, ptr %v529, align 4
  %v530 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 54
  store float 0.0, ptr %v530, align 4
  %v531 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 55
  store float 0.0, ptr %v531, align 4
  %v532 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 56
  store float 0.0, ptr %v532, align 4
  %v533 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 57
  store float 0.0, ptr %v533, align 4
  %v534 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 58
  store float 0.0, ptr %v534, align 4
  %v535 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 59
  store float 0.0, ptr %v535, align 4
  %v536 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 60
  store float 0.0, ptr %v536, align 4
  %v537 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 61
  store float 0.0, ptr %v537, align 4
  %v538 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 62
  store float 0.0, ptr %v538, align 4
  %v539 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 63
  store float 0.0, ptr %v539, align 4
  %v540 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 64
  store float 0.0, ptr %v540, align 4
  %v541 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 65
  store float 0.0, ptr %v541, align 4
  %v542 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 66
  store float 0.0, ptr %v542, align 4
  %v543 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 67
  store float 0.0, ptr %v543, align 4
  %v544 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 68
  store float 0.0, ptr %v544, align 4
  %v545 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 69
  store float 0.0, ptr %v545, align 4
  %v546 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 70
  store float 0.0, ptr %v546, align 4
  %v547 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 71
  store float 0.0, ptr %v547, align 4
  %v548 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 72
  store float 0.0, ptr %v548, align 4
  %v549 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 73
  store float 0.0, ptr %v549, align 4
  %v550 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 74
  store float 0.0, ptr %v550, align 4
  %v551 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 75
  store float 0.0, ptr %v551, align 4
  %v552 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 76
  store float 0.0, ptr %v552, align 4
  %v553 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 77
  store float 0.0, ptr %v553, align 4
  %v554 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 78
  store float 0.0, ptr %v554, align 4
  %v555 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 79
  store float 0.0, ptr %v555, align 4
  %v556 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 80
  store float 0.0, ptr %v556, align 4
  %v557 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 81
  store float 0.0, ptr %v557, align 4
  %v558 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 82
  store float 0.0, ptr %v558, align 4
  %v559 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 83
  store float 0.0, ptr %v559, align 4
  %v560 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 84
  store float 0.0, ptr %v560, align 4
  %v561 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 85
  store float 0.0, ptr %v561, align 4
  %v562 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 86
  store float 0.0, ptr %v562, align 4
  %v563 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 87
  store float 0.0, ptr %v563, align 4
  %v564 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 88
  store float 0.0, ptr %v564, align 4
  %v565 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 89
  store float 0.0, ptr %v565, align 4
  %v566 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 90
  store float 0.0, ptr %v566, align 4
  %v567 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 91
  store float 0.0, ptr %v567, align 4
  %v568 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 92
  store float 0.0, ptr %v568, align 4
  %v569 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 93
  store float 0.0, ptr %v569, align 4
  %v570 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 94
  store float 0.0, ptr %v570, align 4
  %v571 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 95
  store float 0.0, ptr %v571, align 4
  %v572 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 96
  store float 0.0, ptr %v572, align 4
  %v573 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 97
  store float 0.0, ptr %v573, align 4
  %v574 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 98
  store float 0.0, ptr %v574, align 4
  %v575 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 99
  store float 0.0, ptr %v575, align 4
  %v576 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 100
  store float 0.0, ptr %v576, align 4
  %v577 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 101
  store float 0.0, ptr %v577, align 4
  %v578 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 102
  store float 0.0, ptr %v578, align 4
  %v579 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 103
  store float 0.0, ptr %v579, align 4
  %v580 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 104
  store float 0.0, ptr %v580, align 4
  %v581 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 105
  store float 0.0, ptr %v581, align 4
  %v582 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 106
  store float 0.0, ptr %v582, align 4
  %v583 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 107
  store float 0.0, ptr %v583, align 4
  %v584 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 108
  store float 0.0, ptr %v584, align 4
  %v585 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 109
  store float 0.0, ptr %v585, align 4
  %v586 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 110
  store float 0.0, ptr %v586, align 4
  %v587 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 111
  store float 0.0, ptr %v587, align 4
  %v588 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 112
  store float 0.0, ptr %v588, align 4
  %v589 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 113
  store float 0.0, ptr %v589, align 4
  %v590 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 114
  store float 0.0, ptr %v590, align 4
  %v591 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 115
  store float 0.0, ptr %v591, align 4
  %v592 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 116
  store float 0.0, ptr %v592, align 4
  %v593 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 117
  store float 0.0, ptr %v593, align 4
  %v594 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 118
  store float 0.0, ptr %v594, align 4
  %v595 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 119
  store float 0.0, ptr %v595, align 4
  %v596 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 120
  store float 0.0, ptr %v596, align 4
  %v597 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 121
  store float 0.0, ptr %v597, align 4
  %v598 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 122
  store float 0.0, ptr %v598, align 4
  %v599 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 123
  store float 0.0, ptr %v599, align 4
  %v600 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 124
  store float 0.0, ptr %v600, align 4
  %v601 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 125
  store float 0.0, ptr %v601, align 4
  %v602 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 126
  store float 0.0, ptr %v602, align 4
  %v603 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 127
  store float 0.0, ptr %v603, align 4
  br label %bb128
bb128:
  %v604 = phi float [ 0.0, %bb127 ], [ %v624, %bb132 ]
  %v605 = phi i64 [ 0, %bb127 ], [ %v1089, %bb132 ]
  %v606 = icmp ult i64 %v605, %v65
  %v607 = xor i1 %v606, 1
  br i1 %v607, label %bb260, label %bb259
bb129:
  %v608 = extractvalue { i64, i64 } %v1088, 1
  %v609 = add i64 %v475, %v608
  %v610 = extractvalue { ptr, i64 } %v41, 1
  %v611 = icmp ult i64 %v609, %v610
  br i1 %v611, label %bb131, label %bb316
bb130:
  %v612 = fadd contract float %v604, 0.0000009999999974752427
  %v613 = call float @__nv_sqrtf(float %v612) #0
  br label %bb263
bb131:
  %v614 = extractvalue { ptr, i64 } %v41, 0
  %v615 = getelementptr inbounds i16, ptr %v614, i64 %v609
  %v616 = load i16, ptr %v615, align 2
  %v617 = zext i16 %v616 to i32
  %v618 = and i32 16, 31
  %v619 = shl i32 %v617, %v618
  %v620 = bitcast i32 %v619 to float
  %v621 = icmp ult i64 %v608, 128
  br i1 %v621, label %bb132, label %bb317
bb132:
  %v622 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 %v608
  store float %v620, ptr %v622, align 4
  %v623 = fmul contract float %v620, %v620
  %v624 = fadd contract float %v604, %v623
  br label %bb128
bb133:
  br label %bb134
bb134:
  %v625 = phi float [ 0.0, %bb133 ], [ %v647, %bb138 ]
  %v626 = phi i64 [ 0, %bb133 ], [ %v1103, %bb138 ]
  %v627 = icmp ult i64 %v626, %v65
  %v628 = xor i1 %v627, 1
  br i1 %v628, label %bb265, label %bb264
bb135:
  %v629 = extractvalue { i64, i64 } %v1102, 1
  %v630 = icmp ult i64 %v629, 128
  br i1 %v630, label %bb137, label %bb318
bb136:
  %v631 = insertvalue { i64, i64, i1, [7 x i8] } undef, i64 0, 0
  %v632 = insertvalue { i64, i64, i1, [7 x i8] } %v631, i64 %v465, 1
  %v633 = insertvalue { i64, i64, i1, [7 x i8] } %v632, i1 0, 2
  store { i64, i64, i1, [7 x i8] } %v633, ptr %v60, align 8
  br label %bb139
bb137:
  %v634 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 %v629
  %v635 = load float, ptr %v634, align 4
  %v636 = fmul contract float %v635, %v1094
  %v637 = fmul contract float %v636, %v1097
  %v638 = mul i64 %v629, %v66
  %v639 = add i64 %v93, %v638
  %v640 = add i64 %v639, %v466
  %v641 = extractvalue { ptr, i64 } %v48, 1
  %v642 = icmp ult i64 %v640, %v641
  br i1 %v642, label %bb138, label %bb319
bb138:
  %v643 = extractvalue { ptr, i64 } %v48, 0
  %v644 = getelementptr inbounds float, ptr %v643, i64 %v640
  %v645 = load float, ptr %v644, align 4
  %v646 = fmul contract float %v637, %v645
  %v647 = fadd contract float %v625, %v646
  br label %bb134
bb139:
  %v648 = phi float [ 0.0, %bb136 ], [ %v720, %bb153 ]
  %v649 = call { i64, i64 } @_RNvXsc_NtNtCsiQ4CSjCKWVc_4core4iter5rangeINtNtNtB9_3ops5range14RangeInclusivejENtB5_26RangeInclusiveIteratorImpl9spec_nextCsgBauY1x2eDL_17infers_kernel_lib(ptr %v60) #0
  br label %bb268
bb140:
  %v650 = extractvalue { i64, i64 } %v649, 1
  br label %bb142
bb141:
  %v651 = fadd contract float %v625, %v648
  %v652 = mul i64 %v473, %v66
  %v653 = mul i64 %v63, %v66
  %v654 = add i64 %v652, %v653
  %v655 = add i64 %v654, %v466
  %v656 = bitcast float %v651 to i32
  %v657 = and i32 16, 31
  %v658 = lshr i32 %v656, %v657
  %v659 = trunc i32 %v658 to i16
  %v660 = extractvalue { ptr, i64 } %v49, 0
  %v661 = getelementptr inbounds i16, ptr %v660, i64 %v655
  store i16 %v659, ptr %v661, align 2
  br label %bb121
bb142:
  %v662 = phi float [ 0.0, %bb140 ], [ %v682, %bb145 ]
  %v663 = phi i64 [ 0, %bb140 ], [ %v1117, %bb145 ]
  %v664 = icmp ult i64 %v663, %v65
  %v665 = xor i1 %v664, 1
  br i1 %v665, label %bb271, label %bb270
bb143:
  %v666 = extractvalue { i64, i64 } %v1116, 1
  %v667 = icmp ult i64 %v666, 128
  br i1 %v667, label %bb145, label %bb320
bb144:
  %v668 = load float, ptr %v1095, align 4
  %v669 = getelementptr inbounds float, ptr %v89, i64 %v650
  %v670 = load float, ptr %v669, align 4
  %v671 = fsub contract float %v668, %v670
  %v672 = call float @__nv_expf(float %v671) #0
  br label %bb146
bb145:
  %v673 = getelementptr inbounds [128 x float], ptr %v59, i32 0, i64 %v666
  %v674 = load float, ptr %v673, align 4
  %v675 = fmul contract float %v674, %v1094
  %v676 = mul i64 %v650, %v65
  %v677 = add i64 %v676, %v666
  %v678 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v677
  %v679 = addrspacecast ptr addrspace(3) %v678 to ptr
  %v680 = load float, ptr %v679, align 4
  %v681 = fmul contract float %v675, %v680
  %v682 = fadd contract float %v662, %v681
  br label %bb142
bb146:
  %v683 = fmul contract float %v662, %v672
  br label %bb147
bb147:
  %v684 = phi float [ 0.0, %bb146 ], [ %v712, %bb150 ]
  %v685 = phi i64 [ 0, %bb146 ], [ %v1127, %bb150 ]
  %v686 = icmp ult i64 %v685, %v100
  %v687 = xor i1 %v686, 1
  br i1 %v687, label %bb275, label %bb274
bb148:
  %v688 = extractvalue { i64, i64 } %v1126, 1
  %v689 = add i64 %v98, %v688
  %v690 = mul i64 %v689, %v472
  %v691 = mul i64 %v690, %v66
  %v692 = mul i64 %v63, %v66
  %v693 = add i64 %v691, %v692
  %v694 = add i64 %v693, %v466
  %v695 = extractvalue { ptr, i64 } %v43, 1
  %v696 = icmp ult i64 %v694, %v695
  br i1 %v696, label %bb150, label %bb321
bb149:
  br label %bb151
bb150:
  %v697 = extractvalue { ptr, i64 } %v43, 0
  %v698 = getelementptr inbounds i16, ptr %v697, i64 %v694
  %v699 = load i16, ptr %v698, align 2
  %v700 = zext i16 %v699 to i32
  %v701 = and i32 16, 31
  %v702 = shl i32 %v700, %v701
  %v703 = bitcast i32 %v702 to float
  %v704 = getelementptr inbounds float, ptr %v90, i64 %v688
  %v705 = load float, ptr %v704, align 4
  %v706 = fmul contract float %v703, %v705
  %v707 = mul i64 %v650, %v64
  %v708 = add i64 %v707, %v688
  %v709 = getelementptr inbounds float, ptr %v85, i64 %v708
  %v710 = load float, ptr %v709, align 4
  %v711 = fmul contract float %v710, %v706
  %v712 = fadd contract float %v684, %v711
  br label %bb147
bb151:
  %v713 = phi float [ 0.0, %bb149 ], [ %v749, %bb158 ]
  %v714 = phi i64 [ 0, %bb149 ], [ %v1137, %bb158 ]
  %v715 = icmp ult i64 %v714, %v65
  %v716 = xor i1 %v715, 1
  br i1 %v716, label %bb279, label %bb278
bb152:
  %v717 = extractvalue { i64, i64 } %v1136, 1
  br label %bb154
bb153:
  %v718 = fsub contract float %v684, %v713
  %v719 = fmul contract float %v683, %v718
  %v720 = fadd contract float %v648, %v719
  br label %bb139
bb154:
  %v721 = phi float [ 0.0, %bb152 ], [ %v744, %bb157 ]
  %v722 = phi i64 [ 0, %bb152 ], [ %v1147, %bb157 ]
  %v723 = icmp ult i64 %v722, %v100
  %v724 = xor i1 %v723, 1
  br i1 %v724, label %bb283, label %bb282
bb155:
  %v725 = extractvalue { i64, i64 } %v1146, 1
  %v726 = mul i64 %v650, %v64
  %v727 = add i64 %v726, %v725
  %v728 = getelementptr inbounds float, ptr %v85, i64 %v727
  %v729 = load float, ptr %v728, align 4
  %v730 = mul i64 %v725, %v65
  %v731 = add i64 %v730, %v717
  %v732 = getelementptr inbounds float, ptr %v81, i64 %v731
  %v733 = load float, ptr %v732, align 4
  %v734 = fmul contract float %v729, %v733
  %v735 = getelementptr inbounds float, ptr %v89, i64 %v725
  %v736 = load float, ptr %v735, align 4
  %v737 = call float @__nv_expf(float %v736) #0
  br label %bb157
bb156:
  %v738 = mul i64 %v717, %v66
  %v739 = add i64 %v93, %v738
  %v740 = add i64 %v739, %v466
  %v741 = extractvalue { ptr, i64 } %v48, 1
  %v742 = icmp ult i64 %v740, %v741
  br i1 %v742, label %bb158, label %bb322
bb157:
  %v743 = fmul contract float %v734, %v737
  %v744 = fadd contract float %v721, %v743
  br label %bb154
bb158:
  %v745 = extractvalue { ptr, i64 } %v48, 0
  %v746 = getelementptr inbounds float, ptr %v745, i64 %v740
  %v747 = load float, ptr %v746, align 4
  %v748 = fmul contract float %v721, %v747
  %v749 = fadd contract float %v713, %v748
  br label %bb151
bb159:
  %v750 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb161
bb160:
  br label %bb121
bb161:
  %v751 = zext i32 %v750 to i64
  %v752 = mul i64 %v65, %v66
  %v753 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb162
bb162:
  %v754 = zext i32 %v753 to i64
  %v755 = add i64 %v752, %v754
  %v756 = sub i64 %v755, 1
  %v757 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb163
bb163:
  %v758 = zext i32 %v757 to i64
  %v759 = icmp eq i64 %v758, 0
  %v760 = xor i1 %v759, 1
  br i1 %v760, label %bb164, label %bb323
bb164:
  %v761 = udiv i64 %v756, %v758
  %v762 = mul i64 %v751, %v761
  %v763 = add i64 %v751, 1
  %v764 = mul i64 %v763, %v761
  br label %bb165
bb165:
  %v765 = phi i64 [ %v762, %bb164 ], [ %v1157, %bb174 ]
  %v766 = icmp ult i64 %v765, %v764
  %v767 = xor i1 %v766, 1
  br i1 %v767, label %bb287, label %bb286
bb166:
  %v768 = extractvalue { i64, i64 } %v1156, 1
  %v769 = icmp uge i64 %v768, %v752
  %v770 = xor i1 %v769, 1
  br i1 %v770, label %bb168, label %bb167
bb167:
  br label %bb188
bb168:
  %v771 = icmp eq i64 %v66, 0
  %v772 = xor i1 %v771, 1
  br i1 %v772, label %bb169, label %bb324
bb169:
  %v773 = udiv i64 %v768, %v66
  %v774 = urem i64 %v768, %v66
  %v775 = sub i64 %v100, 1
  %v776 = getelementptr inbounds float, ptr %v89, i64 %v775
  %v777 = load float, ptr %v776, align 4
  %v778 = call float @__nv_expf(float %v777) #0
  br label %bb170
bb170:
  %v779 = mul i64 %v773, %v66
  %v780 = add i64 %v93, %v779
  %v781 = add i64 %v780, %v774
  %v782 = extractvalue { ptr, i64 } %v48, 1
  %v783 = icmp ult i64 %v781, %v782
  br i1 %v783, label %bb171, label %bb325
bb171:
  %v784 = extractvalue { ptr, i64 } %v48, 0
  %v785 = getelementptr inbounds float, ptr %v784, i64 %v781
  %v786 = load float, ptr %v785, align 4
  %v787 = fmul contract float %v786, %v778
  br label %bb172
bb172:
  %v788 = phi float [ %v787, %bb171 ], [ %v843, %bb182 ]
  %v789 = phi i64 [ 0, %bb171 ], [ %v1167, %bb182 ]
  %v790 = icmp ult i64 %v789, %v100
  %v791 = xor i1 %v790, 1
  br i1 %v791, label %bb291, label %bb290
bb173:
  %v792 = extractvalue { i64, i64 } %v1166, 1
  %v793 = load float, ptr %v776, align 4
  %v794 = getelementptr inbounds float, ptr %v89, i64 %v792
  %v795 = load float, ptr %v794, align 4
  %v796 = fsub contract float %v793, %v795
  %v797 = call float @__nv_expf(float %v796) #0
  br label %bb175
bb174:
  %v798 = extractvalue { ptr, i64 } %v48, 0
  %v799 = getelementptr inbounds float, ptr %v798, i64 %v781
  store float %v788, ptr %v799, align 4
  br label %bb165
bb175:
  br label %bb176
bb176:
  %v800 = phi float [ 0.0, %bb175 ], [ %v829, %bb179 ]
  %v801 = phi i64 [ 0, %bb175 ], [ %v1177, %bb179 ]
  %v802 = icmp ult i64 %v801, %v100
  %v803 = xor i1 %v802, 1
  br i1 %v803, label %bb295, label %bb294
bb177:
  %v804 = extractvalue { i64, i64 } %v1176, 1
  %v805 = add i64 %v98, %v804
  %v806 = zext i32 %v51 to i64
  %v807 = mul i64 %v805, %v806
  %v808 = mul i64 %v807, %v66
  %v809 = mul i64 %v63, %v66
  %v810 = add i64 %v808, %v809
  %v811 = add i64 %v810, %v774
  %v812 = extractvalue { ptr, i64 } %v43, 1
  %v813 = icmp ult i64 %v811, %v812
  br i1 %v813, label %bb179, label %bb326
bb178:
  br label %bb180
bb179:
  %v814 = extractvalue { ptr, i64 } %v43, 0
  %v815 = getelementptr inbounds i16, ptr %v814, i64 %v811
  %v816 = load i16, ptr %v815, align 2
  %v817 = zext i16 %v816 to i32
  %v818 = and i32 16, 31
  %v819 = shl i32 %v817, %v818
  %v820 = bitcast i32 %v819 to float
  %v821 = getelementptr inbounds float, ptr %v90, i64 %v804
  %v822 = load float, ptr %v821, align 4
  %v823 = fmul contract float %v820, %v822
  %v824 = mul i64 %v792, %v64
  %v825 = add i64 %v824, %v804
  %v826 = getelementptr inbounds float, ptr %v85, i64 %v825
  %v827 = load float, ptr %v826, align 4
  %v828 = fmul contract float %v827, %v823
  %v829 = fadd contract float %v800, %v828
  br label %bb176
bb180:
  %v830 = phi float [ 0.0, %bb178 ], [ %v871, %bb187 ]
  %v831 = phi i64 [ 0, %bb178 ], [ %v1187, %bb187 ]
  %v832 = icmp ult i64 %v831, %v65
  %v833 = xor i1 %v832, 1
  br i1 %v833, label %bb299, label %bb298
bb181:
  %v834 = extractvalue { i64, i64 } %v1186, 1
  br label %bb183
bb182:
  %v835 = fsub contract float %v800, %v830
  %v836 = mul i64 %v792, %v65
  %v837 = add i64 %v836, %v773
  %v838 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_chunked_gated_delta_prefill_bf16, i64 %v837
  %v839 = addrspacecast ptr addrspace(3) %v838 to ptr
  %v840 = load float, ptr %v839, align 4
  %v841 = fmul contract float %v797, %v840
  %v842 = fmul contract float %v841, %v835
  %v843 = fadd contract float %v788, %v842
  br label %bb172
bb183:
  %v844 = phi float [ 0.0, %bb181 ], [ %v866, %bb186 ]
  %v845 = phi i64 [ 0, %bb181 ], [ %v1197, %bb186 ]
  %v846 = icmp ult i64 %v845, %v100
  %v847 = xor i1 %v846, 1
  br i1 %v847, label %bb303, label %bb302
bb184:
  %v848 = extractvalue { i64, i64 } %v1196, 1
  %v849 = mul i64 %v792, %v64
  %v850 = add i64 %v849, %v848
  %v851 = getelementptr inbounds float, ptr %v85, i64 %v850
  %v852 = load float, ptr %v851, align 4
  %v853 = mul i64 %v848, %v65
  %v854 = add i64 %v853, %v834
  %v855 = getelementptr inbounds float, ptr %v81, i64 %v854
  %v856 = load float, ptr %v855, align 4
  %v857 = fmul contract float %v852, %v856
  %v858 = getelementptr inbounds float, ptr %v89, i64 %v848
  %v859 = load float, ptr %v858, align 4
  %v860 = call float @__nv_expf(float %v859) #0
  br label %bb186
bb185:
  %v861 = mul i64 %v834, %v66
  %v862 = add i64 %v93, %v861
  %v863 = add i64 %v862, %v774
  %v864 = icmp ult i64 %v863, %v782
  br i1 %v864, label %bb187, label %bb327
bb186:
  %v865 = fmul contract float %v857, %v860
  %v866 = fadd contract float %v844, %v865
  br label %bb183
bb187:
  %v867 = extractvalue { ptr, i64 } %v48, 0
  %v868 = getelementptr inbounds float, ptr %v867, i64 %v863
  %v869 = load float, ptr %v868, align 4
  %v870 = fmul contract float %v844, %v869
  %v871 = fadd contract float %v830, %v870
  br label %bb180
bb188:
  br label %bb7
bb189:
  %v872 = fdiv contract float 1.0, %v74
  %v873 = extractvalue { ptr, i64 } %v46, 1
  %v874 = icmp ult i64 %v63, %v873
  br i1 %v874, label %bb4, label %bb328
bb190:
  %v875 = add i64 %v94, 1
  %v876 = insertvalue { i64, i64 } undef, i64 1, 0
  %v877 = insertvalue { i64, i64 } %v876, i64 %v94, 1
  br label %bb192
bb191:
  %v878 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb192
bb192:
  %v879 = phi { i64, i64 } [ %v877, %bb190 ], [ %v878, %bb191 ]
  %v880 = phi i64 [ %v875, %bb190 ], [ %v94, %bb191 ]
  %v881 = extractvalue { i64, i64 } %v879, 0
  %v882 = bitcast i64 %v881 to i64
  %v883 = icmp eq i64 %v882, 0
  br i1 %v883, label %bb10, label %bb193
bb193:
  %v884 = icmp eq i64 %v882, 1
  br i1 %v884, label %bb9, label %bb8
bb194:
  %v885 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v55, i32 0, i32 0
  %v886 = getelementptr inbounds { i64, i64 }, ptr %v885, i32 0, i32 0
  %v887 = load i64, ptr %v886, align 8
  %v888 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v55, i32 0, i32 0
  %v889 = getelementptr inbounds { i64, i64 }, ptr %v888, i32 0, i32 1
  %v890 = load i64, ptr %v889, align 8
  %v891 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v55, i32 0, i32 1
  %v892 = load i64, ptr %v891, align 8
  %v893 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v55, i32 0, i32 2
  %v894 = load i1, ptr %v893, align 1
  br label %bb14
bb195:
  %v895 = add i64 %v110, %v122
  %v896 = sub i64 %v111, 1
  %v897 = insertvalue { i64, i64 } undef, i64 1, 0
  %v898 = insertvalue { i64, i64 } %v897, i64 %v110, 1
  br label %bb197
bb196:
  %v899 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb197
bb197:
  %v900 = phi { i64, i64 } [ %v898, %bb195 ], [ %v899, %bb196 ]
  %v901 = phi i64 [ %v895, %bb195 ], [ %v110, %bb196 ]
  %v902 = phi i64 [ %v896, %bb195 ], [ %v111, %bb196 ]
  %v903 = extractvalue { i64, i64 } %v900, 0
  %v904 = bitcast i64 %v903 to i64
  %v905 = icmp eq i64 %v904, 0
  br i1 %v905, label %bb16, label %bb198
bb198:
  %v906 = icmp eq i64 %v904, 1
  br i1 %v906, label %bb15, label %bb8
bb199:
  %v907 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v56, i32 0, i32 0
  %v908 = getelementptr inbounds { i64, i64 }, ptr %v907, i32 0, i32 0
  %v909 = load i64, ptr %v908, align 8
  %v910 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v56, i32 0, i32 0
  %v911 = getelementptr inbounds { i64, i64 }, ptr %v910, i32 0, i32 1
  %v912 = load i64, ptr %v911, align 8
  %v913 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v56, i32 0, i32 1
  %v914 = load i64, ptr %v913, align 8
  %v915 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v56, i32 0, i32 2
  %v916 = load i1, ptr %v915, align 1
  br label %bb34
bb200:
  %v917 = add i64 %v184, %v196
  %v918 = sub i64 %v185, 1
  %v919 = insertvalue { i64, i64 } undef, i64 1, 0
  %v920 = insertvalue { i64, i64 } %v919, i64 %v184, 1
  br label %bb202
bb201:
  %v921 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb202
bb202:
  %v922 = phi { i64, i64 } [ %v920, %bb200 ], [ %v921, %bb201 ]
  %v923 = phi i64 [ %v917, %bb200 ], [ %v184, %bb201 ]
  %v924 = phi i64 [ %v918, %bb200 ], [ %v185, %bb201 ]
  %v925 = extractvalue { i64, i64 } %v922, 0
  %v926 = bitcast i64 %v925 to i64
  %v927 = icmp eq i64 %v926, 0
  br i1 %v927, label %bb36, label %bb203
bb203:
  %v928 = icmp eq i64 %v926, 1
  br i1 %v928, label %bb35, label %bb8
bb204:
  %v929 = add i64 %v208, 1
  %v930 = insertvalue { i64, i64 } undef, i64 1, 0
  %v931 = insertvalue { i64, i64 } %v930, i64 %v208, 1
  br label %bb206
bb205:
  %v932 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb206
bb206:
  %v933 = phi { i64, i64 } [ %v931, %bb204 ], [ %v932, %bb205 ]
  %v934 = phi i64 [ %v929, %bb204 ], [ %v208, %bb205 ]
  %v935 = extractvalue { i64, i64 } %v933, 0
  %v936 = bitcast i64 %v935 to i64
  %v937 = icmp eq i64 %v936, 0
  br i1 %v937, label %bb44, label %bb207
bb207:
  %v938 = icmp eq i64 %v936, 1
  br i1 %v938, label %bb43, label %bb8
bb208:
  %v939 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v57, i32 0, i32 0
  %v940 = getelementptr inbounds { i64, i64 }, ptr %v939, i32 0, i32 0
  %v941 = load i64, ptr %v940, align 8
  %v942 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v57, i32 0, i32 0
  %v943 = getelementptr inbounds { i64, i64 }, ptr %v942, i32 0, i32 1
  %v944 = load i64, ptr %v943, align 8
  %v945 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v57, i32 0, i32 1
  %v946 = load i64, ptr %v945, align 8
  %v947 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v57, i32 0, i32 2
  %v948 = load i1, ptr %v947, align 1
  br label %bb50
bb209:
  %v949 = add i64 %v225, %v237
  %v950 = sub i64 %v226, 1
  %v951 = insertvalue { i64, i64 } undef, i64 1, 0
  %v952 = insertvalue { i64, i64 } %v951, i64 %v225, 1
  br label %bb211
bb210:
  %v953 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb211
bb211:
  %v954 = phi { i64, i64 } [ %v952, %bb209 ], [ %v953, %bb210 ]
  %v955 = phi i64 [ %v949, %bb209 ], [ %v225, %bb210 ]
  %v956 = phi i64 [ %v950, %bb209 ], [ %v226, %bb210 ]
  %v957 = extractvalue { i64, i64 } %v954, 0
  %v958 = bitcast i64 %v957 to i64
  %v959 = icmp eq i64 %v958, 0
  br i1 %v959, label %bb52, label %bb212
bb212:
  %v960 = icmp eq i64 %v958, 1
  br i1 %v960, label %bb51, label %bb8
bb213:
  %v961 = add i64 %v276, 1
  %v962 = insertvalue { i64, i64 } undef, i64 1, 0
  %v963 = insertvalue { i64, i64 } %v962, i64 %v276, 1
  br label %bb215
bb214:
  %v964 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb215
bb215:
  %v965 = phi { i64, i64 } [ %v963, %bb213 ], [ %v964, %bb214 ]
  %v966 = phi i64 [ %v961, %bb213 ], [ %v276, %bb214 ]
  %v967 = extractvalue { i64, i64 } %v965, 0
  %v968 = bitcast i64 %v967 to i64
  %v969 = icmp eq i64 %v968, 0
  br i1 %v969, label %bb63, label %bb216
bb216:
  %v970 = icmp eq i64 %v968, 1
  br i1 %v970, label %bb62, label %bb8
bb217:
  %v971 = fdiv contract float 1.0, %v288
  br label %bb64
bb218:
  %v972 = add i64 %v289, 1
  %v973 = insertvalue { i64, i64 } undef, i64 1, 0
  %v974 = insertvalue { i64, i64 } %v973, i64 %v289, 1
  br label %bb220
bb219:
  %v975 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb220
bb220:
  %v976 = phi { i64, i64 } [ %v974, %bb218 ], [ %v975, %bb219 ]
  %v977 = phi i64 [ %v972, %bb218 ], [ %v289, %bb219 ]
  %v978 = extractvalue { i64, i64 } %v976, 0
  %v979 = bitcast i64 %v978 to i64
  %v980 = icmp eq i64 %v979, 0
  br i1 %v980, label %bb66, label %bb221
bb221:
  %v981 = icmp eq i64 %v979, 1
  br i1 %v981, label %bb65, label %bb8
bb222:
  %v982 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v58, i32 0, i32 0
  %v983 = getelementptr inbounds { i64, i64 }, ptr %v982, i32 0, i32 0
  %v984 = load i64, ptr %v983, align 8
  %v985 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v58, i32 0, i32 0
  %v986 = getelementptr inbounds { i64, i64 }, ptr %v985, i32 0, i32 1
  %v987 = load i64, ptr %v986, align 8
  %v988 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v58, i32 0, i32 1
  %v989 = load i64, ptr %v988, align 8
  %v990 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v58, i32 0, i32 2
  %v991 = load i1, ptr %v990, align 1
  br label %bb70
bb223:
  %v992 = add i64 %v305, %v317
  %v993 = sub i64 %v306, 1
  %v994 = insertvalue { i64, i64 } undef, i64 1, 0
  %v995 = insertvalue { i64, i64 } %v994, i64 %v305, 1
  br label %bb225
bb224:
  %v996 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb225
bb225:
  %v997 = phi { i64, i64 } [ %v995, %bb223 ], [ %v996, %bb224 ]
  %v998 = phi i64 [ %v992, %bb223 ], [ %v305, %bb224 ]
  %v999 = phi i64 [ %v993, %bb223 ], [ %v306, %bb224 ]
  %v1000 = extractvalue { i64, i64 } %v997, 0
  %v1001 = bitcast i64 %v1000 to i64
  %v1002 = icmp eq i64 %v1001, 0
  br i1 %v1002, label %bb72, label %bb226
bb226:
  %v1003 = icmp eq i64 %v1001, 1
  br i1 %v1003, label %bb71, label %bb8
bb227:
  %v1004 = add i64 %v355, 1
  %v1005 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1006 = insertvalue { i64, i64 } %v1005, i64 %v355, 1
  br label %bb229
bb228:
  %v1007 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb229
bb229:
  %v1008 = phi { i64, i64 } [ %v1006, %bb227 ], [ %v1007, %bb228 ]
  %v1009 = phi i64 [ %v1004, %bb227 ], [ %v355, %bb228 ]
  %v1010 = extractvalue { i64, i64 } %v1008, 0
  %v1011 = bitcast i64 %v1010 to i64
  %v1012 = icmp eq i64 %v1011, 0
  br i1 %v1012, label %bb95, label %bb230
bb230:
  %v1013 = icmp eq i64 %v1011, 1
  br i1 %v1013, label %bb85, label %bb8
bb231:
  %v1014 = add i64 %v364, 1
  %v1015 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1016 = insertvalue { i64, i64 } %v1015, i64 %v364, 1
  br label %bb233
bb232:
  %v1017 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb233
bb233:
  %v1018 = phi { i64, i64 } [ %v1016, %bb231 ], [ %v1017, %bb232 ]
  %v1019 = phi i64 [ %v1014, %bb231 ], [ %v364, %bb232 ]
  %v1020 = extractvalue { i64, i64 } %v1018, 0
  %v1021 = bitcast i64 %v1020 to i64
  %v1022 = icmp eq i64 %v1021, 0
  br i1 %v1022, label %bb90, label %bb234
bb234:
  %v1023 = icmp eq i64 %v1021, 1
  br i1 %v1023, label %bb89, label %bb8
bb235:
  %v1024 = add i64 %v396, 1
  %v1025 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1026 = insertvalue { i64, i64 } %v1025, i64 %v396, 1
  br label %bb237
bb236:
  %v1027 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb237
bb237:
  %v1028 = phi { i64, i64 } [ %v1026, %bb235 ], [ %v1027, %bb236 ]
  %v1029 = phi i64 [ %v1024, %bb235 ], [ %v396, %bb236 ]
  %v1030 = extractvalue { i64, i64 } %v1028, 0
  %v1031 = bitcast i64 %v1030 to i64
  %v1032 = icmp eq i64 %v1031, 0
  br i1 %v1032, label %bb101, label %bb238
bb238:
  %v1033 = icmp eq i64 %v1031, 1
  br i1 %v1033, label %bb100, label %bb8
bb239:
  %v1034 = add i64 %v400, 1
  %v1035 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1036 = insertvalue { i64, i64 } %v1035, i64 %v400, 1
  br label %bb241
bb240:
  %v1037 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb241
bb241:
  %v1038 = phi { i64, i64 } [ %v1036, %bb239 ], [ %v1037, %bb240 ]
  %v1039 = phi i64 [ %v1034, %bb239 ], [ %v400, %bb240 ]
  %v1040 = extractvalue { i64, i64 } %v1038, 0
  %v1041 = bitcast i64 %v1040 to i64
  %v1042 = icmp eq i64 %v1041, 0
  br i1 %v1042, label %bb104, label %bb242
bb242:
  %v1043 = icmp eq i64 %v1041, 1
  br i1 %v1043, label %bb103, label %bb8
bb243:
  %v1044 = add i64 %v409, 1
  %v1045 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1046 = insertvalue { i64, i64 } %v1045, i64 %v409, 1
  br label %bb245
bb244:
  %v1047 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb245
bb245:
  %v1048 = phi { i64, i64 } [ %v1046, %bb243 ], [ %v1047, %bb244 ]
  %v1049 = phi i64 [ %v1044, %bb243 ], [ %v409, %bb244 ]
  %v1050 = extractvalue { i64, i64 } %v1048, 0
  %v1051 = bitcast i64 %v1050 to i64
  %v1052 = icmp eq i64 %v1051, 0
  br i1 %v1052, label %bb107, label %bb246
bb246:
  %v1053 = icmp eq i64 %v1051, 1
  br i1 %v1053, label %bb106, label %bb8
bb247:
  %v1054 = add i64 %v414, 1
  %v1055 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1056 = insertvalue { i64, i64 } %v1055, i64 %v414, 1
  br label %bb249
bb248:
  %v1057 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb249
bb249:
  %v1058 = phi { i64, i64 } [ %v1056, %bb247 ], [ %v1057, %bb248 ]
  %v1059 = phi i64 [ %v1054, %bb247 ], [ %v414, %bb248 ]
  %v1060 = extractvalue { i64, i64 } %v1058, 0
  %v1061 = bitcast i64 %v1060 to i64
  %v1062 = icmp eq i64 %v1061, 0
  br i1 %v1062, label %bb110, label %bb250
bb250:
  %v1063 = icmp eq i64 %v1061, 1
  br i1 %v1063, label %bb109, label %bb8
bb251:
  %v1064 = add i64 %v432, 1
  %v1065 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1066 = insertvalue { i64, i64 } %v1065, i64 %v432, 1
  br label %bb253
bb252:
  %v1067 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb253
bb253:
  %v1068 = phi { i64, i64 } [ %v1066, %bb251 ], [ %v1067, %bb252 ]
  %v1069 = phi i64 [ %v1064, %bb251 ], [ %v432, %bb252 ]
  %v1070 = extractvalue { i64, i64 } %v1068, 0
  %v1071 = bitcast i64 %v1070 to i64
  %v1072 = icmp eq i64 %v1071, 0
  br i1 %v1072, label %bb113, label %bb254
bb254:
  %v1073 = icmp eq i64 %v1071, 1
  br i1 %v1073, label %bb112, label %bb8
bb255:
  %v1074 = add i64 %v457, 1
  %v1075 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1076 = insertvalue { i64, i64 } %v1075, i64 %v457, 1
  br label %bb257
bb256:
  %v1077 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb257
bb257:
  %v1078 = phi { i64, i64 } [ %v1076, %bb255 ], [ %v1077, %bb256 ]
  %v1079 = phi i64 [ %v1074, %bb255 ], [ %v457, %bb256 ]
  %v1080 = extractvalue { i64, i64 } %v1078, 0
  %v1081 = bitcast i64 %v1080 to i64
  %v1082 = icmp eq i64 %v1081, 0
  br i1 %v1082, label %bb159, label %bb258
bb258:
  %v1083 = icmp eq i64 %v1081, 1
  br i1 %v1083, label %bb122, label %bb8
bb259:
  %v1084 = add i64 %v605, 1
  %v1085 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1086 = insertvalue { i64, i64 } %v1085, i64 %v605, 1
  br label %bb261
bb260:
  %v1087 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb261
bb261:
  %v1088 = phi { i64, i64 } [ %v1086, %bb259 ], [ %v1087, %bb260 ]
  %v1089 = phi i64 [ %v1084, %bb259 ], [ %v605, %bb260 ]
  %v1090 = extractvalue { i64, i64 } %v1088, 0
  %v1091 = bitcast i64 %v1090 to i64
  %v1092 = icmp eq i64 %v1091, 0
  br i1 %v1092, label %bb130, label %bb262
bb262:
  %v1093 = icmp eq i64 %v1091, 1
  br i1 %v1093, label %bb129, label %bb8
bb263:
  %v1094 = fdiv contract float 1.0, %v613
  %v1095 = getelementptr inbounds float, ptr %v89, i64 %v465
  %v1096 = load float, ptr %v1095, align 4
  %v1097 = call float @__nv_expf(float %v1096) #0
  br label %bb133
bb264:
  %v1098 = add i64 %v626, 1
  %v1099 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1100 = insertvalue { i64, i64 } %v1099, i64 %v626, 1
  br label %bb266
bb265:
  %v1101 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb266
bb266:
  %v1102 = phi { i64, i64 } [ %v1100, %bb264 ], [ %v1101, %bb265 ]
  %v1103 = phi i64 [ %v1098, %bb264 ], [ %v626, %bb265 ]
  %v1104 = extractvalue { i64, i64 } %v1102, 0
  %v1105 = bitcast i64 %v1104 to i64
  %v1106 = icmp eq i64 %v1105, 0
  br i1 %v1106, label %bb136, label %bb267
bb267:
  %v1107 = icmp eq i64 %v1105, 1
  br i1 %v1107, label %bb135, label %bb8
bb268:
  %v1108 = extractvalue { i64, i64 } %v649, 0
  %v1109 = bitcast i64 %v1108 to i64
  %v1110 = icmp eq i64 %v1109, 0
  br i1 %v1110, label %bb141, label %bb269
bb269:
  %v1111 = icmp eq i64 %v1109, 1
  br i1 %v1111, label %bb140, label %bb8
bb270:
  %v1112 = add i64 %v663, 1
  %v1113 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1114 = insertvalue { i64, i64 } %v1113, i64 %v663, 1
  br label %bb272
bb271:
  %v1115 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb272
bb272:
  %v1116 = phi { i64, i64 } [ %v1114, %bb270 ], [ %v1115, %bb271 ]
  %v1117 = phi i64 [ %v1112, %bb270 ], [ %v663, %bb271 ]
  %v1118 = extractvalue { i64, i64 } %v1116, 0
  %v1119 = bitcast i64 %v1118 to i64
  %v1120 = icmp eq i64 %v1119, 0
  br i1 %v1120, label %bb144, label %bb273
bb273:
  %v1121 = icmp eq i64 %v1119, 1
  br i1 %v1121, label %bb143, label %bb8
bb274:
  %v1122 = add i64 %v685, 1
  %v1123 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1124 = insertvalue { i64, i64 } %v1123, i64 %v685, 1
  br label %bb276
bb275:
  %v1125 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb276
bb276:
  %v1126 = phi { i64, i64 } [ %v1124, %bb274 ], [ %v1125, %bb275 ]
  %v1127 = phi i64 [ %v1122, %bb274 ], [ %v685, %bb275 ]
  %v1128 = extractvalue { i64, i64 } %v1126, 0
  %v1129 = bitcast i64 %v1128 to i64
  %v1130 = icmp eq i64 %v1129, 0
  br i1 %v1130, label %bb149, label %bb277
bb277:
  %v1131 = icmp eq i64 %v1129, 1
  br i1 %v1131, label %bb148, label %bb8
bb278:
  %v1132 = add i64 %v714, 1
  %v1133 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1134 = insertvalue { i64, i64 } %v1133, i64 %v714, 1
  br label %bb280
bb279:
  %v1135 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb280
bb280:
  %v1136 = phi { i64, i64 } [ %v1134, %bb278 ], [ %v1135, %bb279 ]
  %v1137 = phi i64 [ %v1132, %bb278 ], [ %v714, %bb279 ]
  %v1138 = extractvalue { i64, i64 } %v1136, 0
  %v1139 = bitcast i64 %v1138 to i64
  %v1140 = icmp eq i64 %v1139, 0
  br i1 %v1140, label %bb153, label %bb281
bb281:
  %v1141 = icmp eq i64 %v1139, 1
  br i1 %v1141, label %bb152, label %bb8
bb282:
  %v1142 = add i64 %v722, 1
  %v1143 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1144 = insertvalue { i64, i64 } %v1143, i64 %v722, 1
  br label %bb284
bb283:
  %v1145 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb284
bb284:
  %v1146 = phi { i64, i64 } [ %v1144, %bb282 ], [ %v1145, %bb283 ]
  %v1147 = phi i64 [ %v1142, %bb282 ], [ %v722, %bb283 ]
  %v1148 = extractvalue { i64, i64 } %v1146, 0
  %v1149 = bitcast i64 %v1148 to i64
  %v1150 = icmp eq i64 %v1149, 0
  br i1 %v1150, label %bb156, label %bb285
bb285:
  %v1151 = icmp eq i64 %v1149, 1
  br i1 %v1151, label %bb155, label %bb8
bb286:
  %v1152 = add i64 %v765, 1
  %v1153 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1154 = insertvalue { i64, i64 } %v1153, i64 %v765, 1
  br label %bb288
bb287:
  %v1155 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb288
bb288:
  %v1156 = phi { i64, i64 } [ %v1154, %bb286 ], [ %v1155, %bb287 ]
  %v1157 = phi i64 [ %v1152, %bb286 ], [ %v765, %bb287 ]
  %v1158 = extractvalue { i64, i64 } %v1156, 0
  %v1159 = bitcast i64 %v1158 to i64
  %v1160 = icmp eq i64 %v1159, 0
  br i1 %v1160, label %bb188, label %bb289
bb289:
  %v1161 = icmp eq i64 %v1159, 1
  br i1 %v1161, label %bb166, label %bb8
bb290:
  %v1162 = add i64 %v789, 1
  %v1163 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1164 = insertvalue { i64, i64 } %v1163, i64 %v789, 1
  br label %bb292
bb291:
  %v1165 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb292
bb292:
  %v1166 = phi { i64, i64 } [ %v1164, %bb290 ], [ %v1165, %bb291 ]
  %v1167 = phi i64 [ %v1162, %bb290 ], [ %v789, %bb291 ]
  %v1168 = extractvalue { i64, i64 } %v1166, 0
  %v1169 = bitcast i64 %v1168 to i64
  %v1170 = icmp eq i64 %v1169, 0
  br i1 %v1170, label %bb174, label %bb293
bb293:
  %v1171 = icmp eq i64 %v1169, 1
  br i1 %v1171, label %bb173, label %bb8
bb294:
  %v1172 = add i64 %v801, 1
  %v1173 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1174 = insertvalue { i64, i64 } %v1173, i64 %v801, 1
  br label %bb296
bb295:
  %v1175 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb296
bb296:
  %v1176 = phi { i64, i64 } [ %v1174, %bb294 ], [ %v1175, %bb295 ]
  %v1177 = phi i64 [ %v1172, %bb294 ], [ %v801, %bb295 ]
  %v1178 = extractvalue { i64, i64 } %v1176, 0
  %v1179 = bitcast i64 %v1178 to i64
  %v1180 = icmp eq i64 %v1179, 0
  br i1 %v1180, label %bb178, label %bb297
bb297:
  %v1181 = icmp eq i64 %v1179, 1
  br i1 %v1181, label %bb177, label %bb8
bb298:
  %v1182 = add i64 %v831, 1
  %v1183 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1184 = insertvalue { i64, i64 } %v1183, i64 %v831, 1
  br label %bb300
bb299:
  %v1185 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb300
bb300:
  %v1186 = phi { i64, i64 } [ %v1184, %bb298 ], [ %v1185, %bb299 ]
  %v1187 = phi i64 [ %v1182, %bb298 ], [ %v831, %bb299 ]
  %v1188 = extractvalue { i64, i64 } %v1186, 0
  %v1189 = bitcast i64 %v1188 to i64
  %v1190 = icmp eq i64 %v1189, 0
  br i1 %v1190, label %bb182, label %bb301
bb301:
  %v1191 = icmp eq i64 %v1189, 1
  br i1 %v1191, label %bb181, label %bb8
bb302:
  %v1192 = add i64 %v845, 1
  %v1193 = insertvalue { i64, i64 } undef, i64 1, 0
  %v1194 = insertvalue { i64, i64 } %v1193, i64 %v845, 1
  br label %bb304
bb303:
  %v1195 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb304
bb304:
  %v1196 = phi { i64, i64 } [ %v1194, %bb302 ], [ %v1195, %bb303 ]
  %v1197 = phi i64 [ %v1192, %bb302 ], [ %v845, %bb303 ]
  %v1198 = extractvalue { i64, i64 } %v1196, 0
  %v1199 = bitcast i64 %v1198 to i64
  %v1200 = icmp eq i64 %v1199, 0
  br i1 %v1200, label %bb185, label %bb305
bb305:
  %v1201 = icmp eq i64 %v1199, 1
  br i1 %v1201, label %bb184, label %bb8
bb306:
  unreachable
bb307:
  unreachable
bb308:
  unreachable
bb309:
  unreachable
bb310:
  unreachable
bb311:
  unreachable
bb312:
  unreachable
bb313:
  unreachable
bb314:
  unreachable
bb315:
  unreachable
bb316:
  unreachable
bb317:
  unreachable
bb318:
  unreachable
bb319:
  unreachable
bb320:
  unreachable
bb321:
  unreachable
bb322:
  unreachable
bb323:
  unreachable
bb324:
  unreachable
bb325:
  unreachable
bb326:
  unreachable
bb327:
  unreachable
bb328:
  unreachable
}

define void @infers_gdn_mamba2_update_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, ptr %v12, i64 %v13, ptr %v14, i64 %v15, i32 %v16, i32 %v17) #0 {
entry:
  %v18 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v1, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v3, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v5, 1
  %v24 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v7, 1
  %v26 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v27 = insertvalue { ptr, i64 } %v26, i64 %v9, 1
  %v28 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v29 = insertvalue { ptr, i64 } %v28, i64 %v11, 1
  %v30 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v31 = insertvalue { ptr, i64 } %v30, i64 %v13, 1
  %v32 = insertvalue { ptr, i64 } undef, ptr %v14, 0
  %v33 = insertvalue { ptr, i64 } %v32, i64 %v15, 1
  br label %bb0
bb0:
  %v34 = phi { ptr, i64 } [ %v19, %entry ]
  %v35 = phi { ptr, i64 } [ %v21, %entry ]
  %v36 = phi { ptr, i64 } [ %v23, %entry ]
  %v37 = phi { ptr, i64 } [ %v25, %entry ]
  %v38 = phi { ptr, i64 } [ %v27, %entry ]
  %v39 = phi { ptr, i64 } [ %v29, %entry ]
  %v40 = phi { ptr, i64 } [ %v31, %entry ]
  %v41 = phi { ptr, i64 } [ %v33, %entry ]
  %v42 = phi i32 [ %v16, %entry ]
  %v43 = phi i32 [ %v17, %entry ]
  %v44 = alloca {  }, align 1
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v46 = mul i32 %v42, %v43
  %v47 = zext i32 %v46 to i64
  %v48 = bitcast ptr %v44 to ptr
  %v49 = call i64 @cuda_device____internal__index_1d(ptr %v48) #0
  br label %bb2
bb2:
  %v50 = icmp uge i64 %v49, %v47
  %v51 = xor i1 %v50, 1
  br i1 %v51, label %bb4, label %bb3
bb3:
  br label %bb27
bb4:
  %v52 = zext i32 %v43 to i64
  %v53 = icmp eq i64 %v52, 0
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb5, label %bb28
bb5:
  %v55 = udiv i64 %v49, %v52
  %v56 = extractvalue { ptr, i64 } %v38, 1
  %v57 = icmp ult i64 %v55, %v56
  br i1 %v57, label %bb6, label %bb29
bb6:
  %v58 = extractvalue { ptr, i64 } %v38, 0
  %v59 = getelementptr inbounds i16, ptr %v58, i64 %v55
  %v60 = load i16, ptr %v59, align 2
  %v61 = zext i16 %v60 to i32
  %v62 = and i32 16, 31
  %v63 = shl i32 %v61, %v62
  %v64 = bitcast i32 %v63 to float
  %v65 = fneg float %v64
  %v66 = call float @__nv_expf(float %v65) #0
  br label %bb7
bb7:
  %v67 = fadd contract float 1.0, %v66
  %v68 = fdiv contract float 1.0, %v67
  %v69 = extractvalue { ptr, i64 } %v39, 1
  %v70 = icmp ult i64 %v55, %v69
  br i1 %v70, label %bb8, label %bb30
bb8:
  %v71 = extractvalue { ptr, i64 } %v39, 0
  %v72 = getelementptr inbounds i16, ptr %v71, i64 %v55
  %v73 = load i16, ptr %v72, align 2
  %v74 = zext i16 %v73 to i32
  %v75 = and i32 16, 31
  %v76 = shl i32 %v74, %v75
  %v77 = bitcast i32 %v76 to float
  %v78 = extractvalue { ptr, i64 } %v36, 1
  %v79 = icmp ult i64 %v49, %v78
  br i1 %v79, label %bb9, label %bb31
bb9:
  %v80 = extractvalue { ptr, i64 } %v36, 0
  %v81 = getelementptr inbounds i16, ptr %v80, i64 %v49
  %v82 = load i16, ptr %v81, align 2
  %v83 = zext i16 %v82 to i32
  %v84 = and i32 16, 31
  %v85 = shl i32 %v83, %v84
  %v86 = bitcast i32 %v85 to float
  %v87 = fadd contract float %v86, %v77
  %v88 = fcmp ogt float %v87, 2.0
  %v89 = xor i1 %v88, 1
  br i1 %v89, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v90 = fcmp olt float %v87, -20.0
  %v91 = xor i1 %v90, 1
  br i1 %v91, label %bb13, label %bb12
bb12:
  br label %bb16
bb13:
  %v92 = call float @__nv_expf(float %v87) #0
  br label %bb14
bb14:
  %v93 = fadd contract float 1.0, %v92
  %v94 = call float @__nv_logf(float %v93) #0
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v95 = phi float [ 0.0, %bb12 ], [ %v94, %bb15 ]
  br label %bb17
bb17:
  %v96 = phi float [ %v87, %bb10 ], [ %v95, %bb16 ]
  %v97 = extractvalue { ptr, i64 } %v35, 1
  %v98 = icmp ult i64 %v55, %v97
  br i1 %v98, label %bb18, label %bb32
bb18:
  %v99 = extractvalue { ptr, i64 } %v35, 0
  %v100 = getelementptr inbounds i16, ptr %v99, i64 %v55
  %v101 = load i16, ptr %v100, align 2
  %v102 = zext i16 %v101 to i32
  %v103 = and i32 16, 31
  %v104 = shl i32 %v102, %v103
  %v105 = bitcast i32 %v104 to float
  %v106 = extractvalue { ptr, i64 } %v40, 1
  %v107 = icmp ult i64 %v49, %v106
  br i1 %v107, label %bb19, label %bb33
bb19:
  %v108 = extractvalue { ptr, i64 } %v40, 0
  %v109 = getelementptr inbounds i16, ptr %v108, i64 %v49
  %v110 = load i16, ptr %v109, align 2
  %v111 = zext i16 %v110 to i32
  %v112 = and i32 16, 31
  %v113 = shl i32 %v111, %v112
  %v114 = bitcast i32 %v113 to float
  %v115 = fmul contract float %v68, %v114
  %v116 = fmul contract float %v96, %v105
  %v117 = fadd contract float %v115, %v116
  %v118 = extractvalue { ptr, i64 } %v34, 1
  %v119 = icmp ult i64 %v55, %v118
  br i1 %v119, label %bb20, label %bb34
bb20:
  %v120 = extractvalue { ptr, i64 } %v34, 0
  %v121 = getelementptr inbounds i16, ptr %v120, i64 %v55
  %v122 = load i16, ptr %v121, align 2
  %v123 = zext i16 %v122 to i32
  %v124 = and i32 16, 31
  %v125 = shl i32 %v123, %v124
  %v126 = bitcast i32 %v125 to float
  %v127 = extractvalue { ptr, i64 } %v37, 1
  %v128 = icmp ult i64 %v49, %v127
  br i1 %v128, label %bb21, label %bb35
bb21:
  %v129 = extractvalue { ptr, i64 } %v37, 0
  %v130 = getelementptr inbounds i16, ptr %v129, i64 %v49
  %v131 = load i16, ptr %v130, align 2
  %v132 = zext i16 %v131 to i32
  %v133 = and i32 16, 31
  %v134 = shl i32 %v132, %v133
  %v135 = bitcast i32 %v134 to float
  %v136 = fcmp ogt float %v135, 0.0
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb24, label %bb22
bb22:
  %v138 = fneg float %v135
  %v139 = call float @__nv_expf(float %v138) #0
  br label %bb23
bb23:
  %v140 = fadd contract float 1.0, %v139
  %v141 = fdiv contract float %v135, %v140
  br label %bb26
bb24:
  %v142 = call float @__nv_expf(float %v135) #0
  br label %bb25
bb25:
  %v143 = fmul contract float %v135, %v142
  %v144 = fadd contract float 1.0, %v142
  %v145 = fdiv contract float %v143, %v144
  br label %bb26
bb26:
  %v146 = phi float [ %v141, %bb23 ], [ %v145, %bb25 ]
  %v147 = fmul contract float %v117, %v126
  %v148 = fmul contract float %v147, %v146
  %v149 = bitcast float %v148 to i32
  %v150 = and i32 16, 31
  %v151 = lshr i32 %v149, %v150
  %v152 = trunc i32 %v151 to i16
  %v153 = extractvalue { ptr, i64 } %v41, 0
  %v154 = getelementptr inbounds i16, ptr %v153, i64 %v49
  store i16 %v152, ptr %v154, align 2
  %v155 = bitcast float %v117 to i32
  %v156 = and i32 16, 31
  %v157 = lshr i32 %v155, %v156
  %v158 = trunc i32 %v157 to i16
  %v159 = extractvalue { ptr, i64 } %v40, 0
  %v160 = getelementptr inbounds i16, ptr %v159, i64 %v49
  store i16 %v158, ptr %v160, align 2
  br label %bb27
bb27:
  ret void
bb28:
  unreachable
bb29:
  unreachable
bb30:
  unreachable
bb31:
  unreachable
bb32:
  unreachable
bb33:
  unreachable
bb34:
  unreachable
bb35:
  unreachable
}

declare float @__nv_fabsf(float)

define void @infers_gdn_gated_delta_prefill_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, ptr %v12, i64 %v13, ptr %v14, i64 %v15, ptr %v16, i64 %v17, i32 %v18, i32 %v19, i32 %v20, i32 %v21) #0 {
entry:
  %v22 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v1, 1
  %v24 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v25 = insertvalue { ptr, i64 } %v24, i64 %v3, 1
  %v26 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v27 = insertvalue { ptr, i64 } %v26, i64 %v5, 1
  %v28 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v29 = insertvalue { ptr, i64 } %v28, i64 %v7, 1
  %v30 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v31 = insertvalue { ptr, i64 } %v30, i64 %v9, 1
  %v32 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v33 = insertvalue { ptr, i64 } %v32, i64 %v11, 1
  %v34 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v35 = insertvalue { ptr, i64 } %v34, i64 %v13, 1
  %v36 = insertvalue { ptr, i64 } undef, ptr %v14, 0
  %v37 = insertvalue { ptr, i64 } %v36, i64 %v15, 1
  %v38 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v39 = insertvalue { ptr, i64 } %v38, i64 %v17, 1
  br label %bb0
bb0:
  %v40 = phi { ptr, i64 } [ %v23, %entry ]
  %v41 = phi { ptr, i64 } [ %v25, %entry ]
  %v42 = phi { ptr, i64 } [ %v27, %entry ]
  %v43 = phi { ptr, i64 } [ %v29, %entry ]
  %v44 = phi { ptr, i64 } [ %v31, %entry ]
  %v45 = phi { ptr, i64 } [ %v33, %entry ]
  %v46 = phi { ptr, i64 } [ %v35, %entry ]
  %v47 = phi { ptr, i64 } [ %v37, %entry ]
  %v48 = phi { ptr, i64 } [ %v39, %entry ]
  %v49 = phi i32 [ %v18, %entry ]
  %v50 = phi i32 [ %v19, %entry ]
  %v51 = phi i32 [ %v20, %entry ]
  %v52 = phi i32 [ %v21, %entry ]
  %v53 = alloca {  }, align 1
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v55 = mul i32 %v50, %v52
  %v56 = zext i32 %v55 to i64
  %v57 = bitcast ptr %v53 to ptr
  %v58 = call i64 @cuda_device____internal__index_1d(ptr %v57) #0
  br label %bb2
bb2:
  %v59 = icmp uge i64 %v58, %v56
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb4, label %bb3
bb3:
  br label %bb69
bb4:
  %v61 = zext i32 %v52 to i64
  %v62 = icmp eq i64 %v61, 0
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb5, label %bb105
bb5:
  %v64 = udiv i64 %v58, %v61
  %v65 = urem i64 %v58, %v61
  %v66 = zext i32 %v51 to i64
  %v67 = zext i32 %v49 to i64
  %v68 = zext i32 %v50 to i64
  %v69 = uitofp i64 %v66 to float
  %v70 = call float @__nv_sqrtf(float %v69) #0
  br label %bb70
bb6:
  %v71 = extractvalue { ptr, i64 } %v45, 0
  %v72 = getelementptr inbounds float, ptr %v71, i64 %v64
  %v73 = load float, ptr %v72, align 4
  %v74 = call float @__nv_expf(float %v73) #0
  br label %bb7
bb7:
  br label %bb8
bb8:
  %v75 = phi i64 [ 0, %bb7 ], [ %v293, %bb63 ]
  %v76 = icmp ult i64 %v75, %v67
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb72, label %bb71
bb9:
  unreachable
bb10:
  %v78 = extractvalue { i64, i64 } %v292, 1
  %v79 = mul i64 %v78, %v68
  %v80 = add i64 %v79, %v64
  %v81 = extractvalue { ptr, i64 } %v43, 1
  %v82 = icmp ult i64 %v80, %v81
  br i1 %v82, label %bb12, label %bb106
bb11:
  br label %bb69
bb12:
  %v83 = extractvalue { ptr, i64 } %v43, 0
  %v84 = getelementptr inbounds i16, ptr %v83, i64 %v80
  %v85 = load i16, ptr %v84, align 2
  %v86 = zext i16 %v85 to i32
  %v87 = and i32 16, 31
  %v88 = shl i32 %v86, %v87
  %v89 = bitcast i32 %v88 to float
  %v90 = extractvalue { ptr, i64 } %v46, 1
  %v91 = icmp ult i64 %v64, %v90
  br i1 %v91, label %bb13, label %bb107
bb13:
  %v92 = extractvalue { ptr, i64 } %v46, 0
  %v93 = getelementptr inbounds float, ptr %v92, i64 %v64
  %v94 = load float, ptr %v93, align 4
  %v95 = fadd contract float %v89, %v94
  %v96 = fcmp ogt float %v95, 20.0
  %v97 = xor i1 %v96, 1
  br i1 %v97, label %bb15, label %bb14
bb14:
  br label %bb21
bb15:
  %v98 = fcmp olt float %v95, -20.0
  %v99 = xor i1 %v98, 1
  br i1 %v99, label %bb17, label %bb16
bb16:
  br label %bb20
bb17:
  %v100 = call float @__nv_expf(float %v95) #0
  br label %bb18
bb18:
  %v101 = fadd contract float 1.0, %v100
  %v102 = call float @__nv_logf(float %v101) #0
  br label %bb19
bb19:
  br label %bb20
bb20:
  %v103 = phi float [ 0.0, %bb16 ], [ %v102, %bb19 ]
  br label %bb21
bb21:
  %v104 = phi float [ %v95, %bb14 ], [ %v103, %bb20 ]
  %v105 = fneg float %v74
  %v106 = fmul contract float %v105, %v104
  %v107 = extractvalue { ptr, i64 } %v44, 1
  %v108 = icmp ult i64 %v80, %v107
  br i1 %v108, label %bb22, label %bb108
bb22:
  %v109 = extractvalue { ptr, i64 } %v44, 0
  %v110 = getelementptr inbounds i16, ptr %v109, i64 %v80
  %v111 = load i16, ptr %v110, align 2
  %v112 = zext i16 %v111 to i32
  %v113 = and i32 16, 31
  %v114 = shl i32 %v112, %v113
  %v115 = bitcast i32 %v114 to float
  %v116 = fneg float %v115
  %v117 = call float @__nv_expf(float %v116) #0
  br label %bb23
bb23:
  %v118 = fadd contract float 1.0, %v117
  %v119 = fdiv contract float 1.0, %v118
  %v120 = call float @__nv_expf(float %v106) #0
  br label %bb24
bb24:
  br label %bb25
bb25:
  %v121 = phi float [ 0.0, %bb24 ], [ %v152, %bb29 ]
  %v122 = phi float [ 0.0, %bb24 ], [ %v154, %bb29 ]
  %v123 = phi i64 [ 0, %bb24 ], [ %v303, %bb29 ]
  %v124 = icmp ult i64 %v123, %v66
  %v125 = xor i1 %v124, 1
  br i1 %v125, label %bb76, label %bb75
bb26:
  %v126 = extractvalue { i64, i64 } %v302, 1
  %v127 = mul i64 %v79, %v66
  %v128 = mul i64 %v64, %v66
  %v129 = add i64 %v127, %v128
  %v130 = add i64 %v129, %v126
  %v131 = extractvalue { ptr, i64 } %v41, 1
  %v132 = icmp ult i64 %v130, %v131
  br i1 %v132, label %bb28, label %bb109
bb27:
  %v133 = fadd contract float %v121, 0.0000009999999974752427
  %v134 = call float @__nv_sqrtf(float %v133) #0
  br label %bb79
bb28:
  %v135 = extractvalue { ptr, i64 } %v41, 0
  %v136 = getelementptr inbounds i16, ptr %v135, i64 %v130
  %v137 = load i16, ptr %v136, align 2
  %v138 = zext i16 %v137 to i32
  %v139 = and i32 16, 31
  %v140 = shl i32 %v138, %v139
  %v141 = bitcast i32 %v140 to float
  %v142 = extractvalue { ptr, i64 } %v40, 1
  %v143 = icmp ult i64 %v130, %v142
  br i1 %v143, label %bb29, label %bb110
bb29:
  %v144 = extractvalue { ptr, i64 } %v40, 0
  %v145 = getelementptr inbounds i16, ptr %v144, i64 %v130
  %v146 = load i16, ptr %v145, align 2
  %v147 = zext i16 %v146 to i32
  %v148 = and i32 16, 31
  %v149 = shl i32 %v147, %v148
  %v150 = bitcast i32 %v149 to float
  %v151 = fmul contract float %v141, %v141
  %v152 = fadd contract float %v121, %v151
  %v153 = fmul contract float %v150, %v150
  %v154 = fadd contract float %v122, %v153
  br label %bb25
bb30:
  %v155 = phi i64 [ %v321, %bb36 ], [ 0, %bb80 ]
  %v156 = icmp ult i64 %v155, %v66
  %v157 = xor i1 %v156, 1
  br i1 %v157, label %bb82, label %bb81
bb31:
  %v158 = extractvalue { i64, i64 } %v320, 1
  %v159 = mul i64 %v158, %v61
  %v160 = add i64 %v315, %v159
  %v161 = extractvalue { ptr, i64 } %v47, 1
  %v162 = icmp ult i64 %v160, %v161
  br i1 %v162, label %bb33, label %bb111
bb32:
  br label %bb37
bb33:
  %v163 = extractvalue { ptr, i64 } %v47, 0
  %v164 = getelementptr inbounds float, ptr %v163, i64 %v160
  %v165 = load float, ptr %v164, align 4
  %v166 = call float @__nv_fabsf(float %v165) #0
  br label %bb85
bb34:
  %v167 = fmul contract float %v165, %v120
  br label %bb36
bb35:
  br label %bb36
bb36:
  %v168 = phi float [ %v167, %bb34 ], [ 0.0, %bb35 ]
  %v169 = extractvalue { ptr, i64 } %v47, 0
  %v170 = getelementptr inbounds float, ptr %v169, i64 %v160
  store float %v168, ptr %v170, align 4
  br label %bb30
bb37:
  %v171 = phi float [ 0.0, %bb32 ], [ %v205, %bb43 ]
  %v172 = phi i64 [ 0, %bb32 ], [ %v333, %bb43 ]
  %v173 = icmp ult i64 %v172, %v66
  %v174 = xor i1 %v173, 1
  br i1 %v174, label %bb87, label %bb86
bb38:
  %v175 = extractvalue { i64, i64 } %v332, 1
  %v176 = mul i64 %v175, %v61
  %v177 = add i64 %v315, %v176
  %v178 = extractvalue { ptr, i64 } %v47, 1
  %v179 = icmp ult i64 %v177, %v178
  br i1 %v179, label %bb40, label %bb112
bb39:
  %v180 = mul i64 %v79, %v61
  %v181 = mul i64 %v64, %v61
  %v182 = add i64 %v180, %v181
  %v183 = add i64 %v182, %v65
  %v184 = extractvalue { ptr, i64 } %v42, 1
  %v185 = icmp ult i64 %v183, %v184
  br i1 %v185, label %bb44, label %bb113
bb40:
  %v186 = extractvalue { ptr, i64 } %v47, 0
  %v187 = getelementptr inbounds float, ptr %v186, i64 %v177
  %v188 = load float, ptr %v187, align 4
  %v189 = mul i64 %v79, %v66
  %v190 = add i64 %v189, %v313
  %v191 = add i64 %v190, %v175
  %v192 = extractvalue { ptr, i64 } %v41, 1
  %v193 = icmp ult i64 %v191, %v192
  br i1 %v193, label %bb41, label %bb114
bb41:
  %v194 = extractvalue { ptr, i64 } %v41, 0
  %v195 = getelementptr inbounds i16, ptr %v194, i64 %v191
  %v196 = load i16, ptr %v195, align 2
  %v197 = zext i16 %v196 to i32
  %v198 = and i32 16, 31
  %v199 = shl i32 %v197, %v198
  %v200 = bitcast i32 %v199 to float
  %v201 = fmul contract float %v200, %v308
  %v202 = call float @__nv_fabsf(float %v201) #0
  br label %bb90
bb42:
  %v203 = fmul contract float %v188, %v201
  %v204 = fadd contract float %v171, %v203
  br label %bb43
bb43:
  %v205 = phi float [ %v204, %bb42 ], [ %v171, %bb90 ]
  br label %bb37
bb44:
  %v206 = extractvalue { ptr, i64 } %v42, 0
  %v207 = getelementptr inbounds i16, ptr %v206, i64 %v183
  %v208 = load i16, ptr %v207, align 2
  %v209 = zext i16 %v208 to i32
  %v210 = and i32 16, 31
  %v211 = shl i32 %v209, %v210
  %v212 = bitcast i32 %v211 to float
  %v213 = call float @__nv_fabsf(float %v212) #0
  br label %bb91
bb45:
  br label %bb47
bb46:
  br label %bb47
bb47:
  %v214 = phi float [ %v212, %bb45 ], [ 0.0, %bb46 ]
  %v215 = call float @__nv_fabsf(float %v119) #0
  br label %bb92
bb48:
  %v216 = fsub contract float %v214, %v171
  %v217 = fmul contract float %v119, %v216
  br label %bb50
bb49:
  br label %bb50
bb50:
  %v218 = phi float [ %v217, %bb48 ], [ 0.0, %bb49 ]
  %v219 = call float @__nv_fabsf(float %v218) #0
  br label %bb93
bb51:
  br label %bb52
bb52:
  %v220 = phi i64 [ 0, %bb51 ], [ %v351, %bb58 ]
  %v221 = icmp ult i64 %v220, %v66
  %v222 = xor i1 %v221, 1
  br i1 %v222, label %bb95, label %bb94
bb53:
  %v223 = extractvalue { i64, i64 } %v350, 1
  %v224 = mul i64 %v79, %v66
  %v225 = add i64 %v224, %v313
  %v226 = add i64 %v225, %v223
  %v227 = extractvalue { ptr, i64 } %v41, 1
  %v228 = icmp ult i64 %v226, %v227
  br i1 %v228, label %bb55, label %bb115
bb54:
  br label %bb60
bb55:
  %v229 = extractvalue { ptr, i64 } %v41, 0
  %v230 = getelementptr inbounds i16, ptr %v229, i64 %v226
  %v231 = load i16, ptr %v230, align 2
  %v232 = zext i16 %v231 to i32
  %v233 = and i32 16, 31
  %v234 = shl i32 %v232, %v233
  %v235 = bitcast i32 %v234 to float
  %v236 = fmul contract float %v235, %v308
  %v237 = call float @__nv_fabsf(float %v236) #0
  br label %bb98
bb56:
  %v238 = fmul contract float %v236, %v218
  %v239 = mul i64 %v223, %v61
  %v240 = add i64 %v315, %v239
  %v241 = extractvalue { ptr, i64 } %v47, 1
  %v242 = icmp ult i64 %v240, %v241
  br i1 %v242, label %bb57, label %bb116
bb57:
  %v243 = extractvalue { ptr, i64 } %v47, 0
  %v244 = getelementptr inbounds float, ptr %v243, i64 %v240
  %v245 = load float, ptr %v244, align 4
  %v246 = fadd contract float %v245, %v238
  %v247 = extractvalue { ptr, i64 } %v47, 0
  %v248 = getelementptr inbounds float, ptr %v247, i64 %v240
  store float %v246, ptr %v248, align 4
  br label %bb58
bb58:
  br label %bb52
bb59:
  br label %bb60
bb60:
  br label %bb61
bb61:
  %v249 = phi float [ 0.0, %bb60 ], [ %v284, %bb68 ]
  %v250 = phi i64 [ 0, %bb60 ], [ %v363, %bb68 ]
  %v251 = icmp ult i64 %v250, %v66
  %v252 = xor i1 %v251, 1
  br i1 %v252, label %bb100, label %bb99
bb62:
  %v253 = extractvalue { i64, i64 } %v362, 1
  %v254 = mul i64 %v253, %v61
  %v255 = add i64 %v315, %v254
  %v256 = extractvalue { ptr, i64 } %v47, 1
  %v257 = icmp ult i64 %v255, %v256
  br i1 %v257, label %bb64, label %bb117
bb63:
  %v258 = bitcast float %v249 to i32
  %v259 = and i32 16, 31
  %v260 = lshr i32 %v258, %v259
  %v261 = trunc i32 %v260 to i16
  %v262 = extractvalue { ptr, i64 } %v48, 0
  %v263 = getelementptr inbounds i16, ptr %v262, i64 %v183
  store i16 %v261, ptr %v263, align 2
  br label %bb8
bb64:
  %v264 = extractvalue { ptr, i64 } %v47, 0
  %v265 = getelementptr inbounds float, ptr %v264, i64 %v255
  %v266 = load float, ptr %v265, align 4
  %v267 = mul i64 %v79, %v66
  %v268 = add i64 %v267, %v313
  %v269 = add i64 %v268, %v253
  %v270 = extractvalue { ptr, i64 } %v40, 1
  %v271 = icmp ult i64 %v269, %v270
  br i1 %v271, label %bb65, label %bb118
bb65:
  %v272 = extractvalue { ptr, i64 } %v40, 0
  %v273 = getelementptr inbounds i16, ptr %v272, i64 %v269
  %v274 = load i16, ptr %v273, align 2
  %v275 = zext i16 %v274 to i32
  %v276 = and i32 16, 31
  %v277 = shl i32 %v275, %v276
  %v278 = bitcast i32 %v277 to float
  %v279 = fmul contract float %v278, %v312
  %v280 = call float @__nv_fabsf(float %v266) #0
  br label %bb103
bb66:
  %v281 = call float @__nv_fabsf(float %v279) #0
  br label %bb104
bb67:
  %v282 = fmul contract float %v266, %v279
  %v283 = fadd contract float %v249, %v282
  br label %bb68
bb68:
  %v284 = phi float [ %v283, %bb67 ], [ %v249, %bb103 ], [ %v249, %bb104 ]
  br label %bb61
bb69:
  ret void
bb70:
  %v285 = fdiv contract float 1.0, %v70
  %v286 = extractvalue { ptr, i64 } %v45, 1
  %v287 = icmp ult i64 %v64, %v286
  br i1 %v287, label %bb6, label %bb119
bb71:
  %v288 = add i64 %v75, 1
  %v289 = insertvalue { i64, i64 } undef, i64 1, 0
  %v290 = insertvalue { i64, i64 } %v289, i64 %v75, 1
  br label %bb73
bb72:
  %v291 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb73
bb73:
  %v292 = phi { i64, i64 } [ %v290, %bb71 ], [ %v291, %bb72 ]
  %v293 = phi i64 [ %v288, %bb71 ], [ %v75, %bb72 ]
  %v294 = extractvalue { i64, i64 } %v292, 0
  %v295 = bitcast i64 %v294 to i64
  %v296 = icmp eq i64 %v295, 0
  br i1 %v296, label %bb11, label %bb74
bb74:
  %v297 = icmp eq i64 %v295, 1
  br i1 %v297, label %bb10, label %bb9
bb75:
  %v298 = add i64 %v123, 1
  %v299 = insertvalue { i64, i64 } undef, i64 1, 0
  %v300 = insertvalue { i64, i64 } %v299, i64 %v123, 1
  br label %bb77
bb76:
  %v301 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb77
bb77:
  %v302 = phi { i64, i64 } [ %v300, %bb75 ], [ %v301, %bb76 ]
  %v303 = phi i64 [ %v298, %bb75 ], [ %v123, %bb76 ]
  %v304 = extractvalue { i64, i64 } %v302, 0
  %v305 = bitcast i64 %v304 to i64
  %v306 = icmp eq i64 %v305, 0
  br i1 %v306, label %bb27, label %bb78
bb78:
  %v307 = icmp eq i64 %v305, 1
  br i1 %v307, label %bb26, label %bb9
bb79:
  %v308 = fdiv contract float 1.0, %v134
  %v309 = fadd contract float %v122, 0.0000009999999974752427
  %v310 = call float @__nv_sqrtf(float %v309) #0
  br label %bb80
bb80:
  %v311 = fdiv contract float 1.0, %v310
  %v312 = fmul contract float %v311, %v285
  %v313 = mul i64 %v64, %v66
  %v314 = mul i64 %v313, %v61
  %v315 = add i64 %v314, %v65
  br label %bb30
bb81:
  %v316 = add i64 %v155, 1
  %v317 = insertvalue { i64, i64 } undef, i64 1, 0
  %v318 = insertvalue { i64, i64 } %v317, i64 %v155, 1
  br label %bb83
bb82:
  %v319 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb83
bb83:
  %v320 = phi { i64, i64 } [ %v318, %bb81 ], [ %v319, %bb82 ]
  %v321 = phi i64 [ %v316, %bb81 ], [ %v155, %bb82 ]
  %v322 = extractvalue { i64, i64 } %v320, 0
  %v323 = bitcast i64 %v322 to i64
  %v324 = icmp eq i64 %v323, 0
  br i1 %v324, label %bb32, label %bb84
bb84:
  %v325 = icmp eq i64 %v323, 1
  br i1 %v325, label %bb31, label %bb9
bb85:
  %v326 = fcmp olt float %v166, 0x7FF0000000000000
  %v327 = xor i1 %v326, 1
  br i1 %v327, label %bb35, label %bb34
bb86:
  %v328 = add i64 %v172, 1
  %v329 = insertvalue { i64, i64 } undef, i64 1, 0
  %v330 = insertvalue { i64, i64 } %v329, i64 %v172, 1
  br label %bb88
bb87:
  %v331 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb88
bb88:
  %v332 = phi { i64, i64 } [ %v330, %bb86 ], [ %v331, %bb87 ]
  %v333 = phi i64 [ %v328, %bb86 ], [ %v172, %bb87 ]
  %v334 = extractvalue { i64, i64 } %v332, 0
  %v335 = bitcast i64 %v334 to i64
  %v336 = icmp eq i64 %v335, 0
  br i1 %v336, label %bb39, label %bb89
bb89:
  %v337 = icmp eq i64 %v335, 1
  br i1 %v337, label %bb38, label %bb9
bb90:
  %v338 = fcmp olt float %v202, 0x7FF0000000000000
  %v339 = xor i1 %v338, 1
  br i1 %v339, label %bb43, label %bb42
bb91:
  %v340 = fcmp olt float %v213, 0x7FF0000000000000
  %v341 = xor i1 %v340, 1
  br i1 %v341, label %bb46, label %bb45
bb92:
  %v342 = fcmp olt float %v215, 0x7FF0000000000000
  %v343 = xor i1 %v342, 1
  br i1 %v343, label %bb49, label %bb48
bb93:
  %v344 = fcmp olt float %v219, 0x7FF0000000000000
  %v345 = xor i1 %v344, 1
  br i1 %v345, label %bb59, label %bb51
bb94:
  %v346 = add i64 %v220, 1
  %v347 = insertvalue { i64, i64 } undef, i64 1, 0
  %v348 = insertvalue { i64, i64 } %v347, i64 %v220, 1
  br label %bb96
bb95:
  %v349 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb96
bb96:
  %v350 = phi { i64, i64 } [ %v348, %bb94 ], [ %v349, %bb95 ]
  %v351 = phi i64 [ %v346, %bb94 ], [ %v220, %bb95 ]
  %v352 = extractvalue { i64, i64 } %v350, 0
  %v353 = bitcast i64 %v352 to i64
  %v354 = icmp eq i64 %v353, 0
  br i1 %v354, label %bb54, label %bb97
bb97:
  %v355 = icmp eq i64 %v353, 1
  br i1 %v355, label %bb53, label %bb9
bb98:
  %v356 = fcmp olt float %v237, 0x7FF0000000000000
  %v357 = xor i1 %v356, 1
  br i1 %v357, label %bb58, label %bb56
bb99:
  %v358 = add i64 %v250, 1
  %v359 = insertvalue { i64, i64 } undef, i64 1, 0
  %v360 = insertvalue { i64, i64 } %v359, i64 %v250, 1
  br label %bb101
bb100:
  %v361 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb101
bb101:
  %v362 = phi { i64, i64 } [ %v360, %bb99 ], [ %v361, %bb100 ]
  %v363 = phi i64 [ %v358, %bb99 ], [ %v250, %bb100 ]
  %v364 = extractvalue { i64, i64 } %v362, 0
  %v365 = bitcast i64 %v364 to i64
  %v366 = icmp eq i64 %v365, 0
  br i1 %v366, label %bb63, label %bb102
bb102:
  %v367 = icmp eq i64 %v365, 1
  br i1 %v367, label %bb62, label %bb9
bb103:
  %v368 = fcmp olt float %v280, 0x7FF0000000000000
  %v369 = xor i1 %v368, 1
  br i1 %v369, label %bb68, label %bb66
bb104:
  %v370 = fcmp olt float %v281, 0x7FF0000000000000
  %v371 = xor i1 %v370, 1
  br i1 %v371, label %bb68, label %bb67
bb105:
  unreachable
bb106:
  unreachable
bb107:
  unreachable
bb108:
  unreachable
bb109:
  unreachable
bb110:
  unreachable
bb111:
  unreachable
bb112:
  unreachable
bb113:
  unreachable
bb114:
  unreachable
bb115:
  unreachable
bb116:
  unreachable
bb117:
  unreachable
bb118:
  unreachable
bb119:
  unreachable
}

define void @infers_gdn_update_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, i32 %v12) #0 {
entry:
  %v13 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v5, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v7, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v9, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v11, 1
  br label %bb0
bb0:
  %v25 = phi { ptr, i64 } [ %v14, %entry ]
  %v26 = phi { ptr, i64 } [ %v16, %entry ]
  %v27 = phi { ptr, i64 } [ %v18, %entry ]
  %v28 = phi { ptr, i64 } [ %v20, %entry ]
  %v29 = phi { ptr, i64 } [ %v22, %entry ]
  %v30 = phi { ptr, i64 } [ %v24, %entry ]
  %v31 = phi i32 [ %v12, %entry ]
  %v32 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v33 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v34 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v37 = zext i32 %v36 to i64
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v39 = zext i32 %v38 to i64
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb4
bb4:
  %v41 = zext i32 %v40 to i64
  %v42 = zext i32 %v31 to i64
  br label %bb5
bb5:
  %v43 = extractvalue { ptr, i64 } %v30, 1
  %v44 = icmp ult i64 %v37, %v43
  br i1 %v44, label %bb6, label %bb65
bb6:
  %v45 = extractvalue { ptr, i64 } %v30, 0
  %v46 = getelementptr inbounds i16, ptr %v45, i64 %v37
  %v47 = load i16, ptr %v46, align 2
  %v48 = zext i16 %v47 to i32
  %v49 = and i32 16, 31
  %v50 = shl i32 %v48, %v49
  %v51 = bitcast i32 %v50 to float
  %v52 = extractvalue { ptr, i64 } %v29, 1
  %v53 = icmp ult i64 %v37, %v52
  br i1 %v53, label %bb7, label %bb66
bb7:
  %v54 = extractvalue { ptr, i64 } %v29, 0
  %v55 = getelementptr inbounds i16, ptr %v54, i64 %v37
  %v56 = load i16, ptr %v55, align 2
  %v57 = zext i16 %v56 to i32
  %v58 = and i32 16, 31
  %v59 = shl i32 %v57, %v58
  %v60 = bitcast i32 %v59 to float
  %v61 = extractvalue { ptr, i64 } %v27, 1
  %v62 = icmp ult i64 %v37, %v61
  br i1 %v62, label %bb8, label %bb67
bb8:
  %v63 = extractvalue { ptr, i64 } %v27, 0
  %v64 = getelementptr inbounds i16, ptr %v63, i64 %v37
  %v65 = load i16, ptr %v64, align 2
  %v66 = zext i16 %v65 to i32
  %v67 = and i32 16, 31
  %v68 = shl i32 %v66, %v67
  %v69 = bitcast i32 %v68 to float
  %v70 = insertvalue { i64, i64 } undef, i64 %v39, 0
  %v71 = insertvalue { i64, i64 } %v70, i64 %v42, 1
  %v72 = extractvalue { i64, i64 } %v71, 0
  %v73 = extractvalue { i64, i64 } %v71, 1
  %v74 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v72, i64 %v73, i64 %v41) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v74, ptr %v32, align 8
  br label %bb50
bb9:
  %v75 = phi float [ %v116, %bb14 ], [ 0.0, %bb50 ]
  %v76 = phi i64 [ %v276, %bb14 ], [ %v262, %bb50 ]
  %v77 = phi i64 [ %v277, %bb14 ], [ %v265, %bb50 ]
  %v78 = add i64 %v267, 1
  %v79 = icmp eq i64 %v78, 0
  %v80 = select i1 %v79, i8 0, i8 1
  %v81 = insertvalue { i8, { { i64 } } } undef, i8 %v80, 0
  %v82 = insertvalue { i8, { { i64 } } } %v81, i64 %v78, 1, 0, 0
  %v83 = extractvalue { i8, { { i64 } } } %v82, 0
  %v84 = zext i8 %v83 to i64
  %v85 = icmp eq i64 %v84, 1
  %v86 = extractvalue { i8, { { i64 } } } %v82, 1
  %v87 = alloca { { i64 } }, align 8
  store { { i64 } } %v86, ptr %v87, align 8
  %v88 = load i64, ptr %v87, align 8
  %v89 = icmp ugt i64 %v77, 0
  %v90 = xor i1 %v89, 1
  br i1 %v90, label %bb52, label %bb51
bb10:
  unreachable
bb11:
  %v91 = extractvalue { i64, i64 } %v275, 1
  %v92 = mul i64 %v37, %v42
  %v93 = add i64 %v92, %v91
  %v94 = extractvalue { ptr, i64 } %v25, 1
  %v95 = icmp ult i64 %v93, %v94
  br i1 %v95, label %bb13, label %bb68
bb12:
  %v96 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_update_bf16, i64 %v39
  %v97 = addrspacecast ptr addrspace(3) %v96 to ptr
  store float %v75, ptr %v97, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb15
bb13:
  %v99 = extractvalue { ptr, i64 } %v25, 0
  %v100 = getelementptr inbounds i16, ptr %v99, i64 %v93
  %v101 = load i16, ptr %v100, align 2
  %v102 = zext i16 %v101 to i32
  %v103 = and i32 16, 31
  %v104 = shl i32 %v102, %v103
  %v105 = bitcast i32 %v104 to float
  %v106 = extractvalue { ptr, i64 } %v28, 1
  %v107 = icmp ult i64 %v91, %v106
  br i1 %v107, label %bb14, label %bb69
bb14:
  %v108 = extractvalue { ptr, i64 } %v28, 0
  %v109 = getelementptr inbounds i16, ptr %v108, i64 %v91
  %v110 = load i16, ptr %v109, align 2
  %v111 = zext i16 %v110 to i32
  %v112 = and i32 16, 31
  %v113 = shl i32 %v111, %v112
  %v114 = bitcast i32 %v113 to float
  %v115 = fmul contract float %v105, %v114
  %v116 = fadd contract float %v75, %v115
  br label %bb9
bb15:
  %v117 = udiv i64 %v41, 2
  br label %bb16
bb16:
  %v118 = phi i64 [ %v117, %bb15 ], [ %v135, %bb23 ]
  %v119 = icmp ugt i64 %v118, 0
  %v120 = xor i1 %v119, 1
  br i1 %v120, label %bb24, label %bb17
bb17:
  call void @llvm.nvvm.barrier0() #0
  br label %bb18
bb18:
  %v122 = icmp ult i64 %v39, %v118
  %v123 = xor i1 %v122, 1
  br i1 %v123, label %bb22, label %bb19
bb19:
  %v124 = add i64 %v39, %v118
  %v125 = icmp ult i64 %v124, %v41
  %v126 = xor i1 %v125, 1
  br i1 %v126, label %bb21, label %bb20
bb20:
  %v127 = add i64 %v39, %v118
  %v128 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_update_bf16, i64 %v127
  %v129 = addrspacecast ptr addrspace(3) %v128 to ptr
  %v130 = load float, ptr %v129, align 4
  %v131 = load float, ptr %v97, align 4
  %v132 = fadd contract float %v131, %v130
  store float %v132, ptr %v97, align 4
  br label %bb23
bb21:
  br label %bb23
bb22:
  br label %bb23
bb23:
  %v133 = zext i32 1 to i64
  %v134 = and i64 %v133, 63
  %v135 = lshr i64 %v118, %v134
  br label %bb16
bb24:
  call void @llvm.nvvm.barrier0() #0
  br label %bb25
bb25:
  %v137 = load float, ptr addrspace(3) @__dynamic_smem_infers_gdn_update_bf16, align 4
  %v138 = fmul contract float %v60, %v69
  %v139 = fmul contract float %v138, %v137
  %v140 = fsub contract float %v51, %v139
  %v141 = extractvalue { i64, i64 } %v71, 0
  %v142 = extractvalue { i64, i64 } %v71, 1
  %v143 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v141, i64 %v142, i64 %v41) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v143, ptr %v33, align 8
  br label %bb55
bb26:
  %v144 = phi i64 [ %v298, %bb30 ], [ %v284, %bb55 ]
  %v145 = phi i64 [ %v299, %bb30 ], [ %v287, %bb55 ]
  %v146 = add i64 %v289, 1
  %v147 = icmp eq i64 %v146, 0
  %v148 = select i1 %v147, i8 0, i8 1
  %v149 = insertvalue { i8, { { i64 } } } undef, i8 %v148, 0
  %v150 = insertvalue { i8, { { i64 } } } %v149, i64 %v146, 1, 0, 0
  %v151 = extractvalue { i8, { { i64 } } } %v150, 0
  %v152 = zext i8 %v151 to i64
  %v153 = icmp eq i64 %v152, 1
  %v154 = extractvalue { i8, { { i64 } } } %v150, 1
  %v155 = alloca { { i64 } }, align 8
  store { { i64 } } %v154, ptr %v155, align 8
  %v156 = load i64, ptr %v155, align 8
  %v157 = icmp ugt i64 %v145, 0
  %v158 = xor i1 %v157, 1
  br i1 %v158, label %bb57, label %bb56
bb27:
  %v159 = extractvalue { i64, i64 } %v297, 1
  %v160 = mul i64 %v37, %v42
  %v161 = add i64 %v160, %v159
  %v162 = extractvalue { ptr, i64 } %v25, 1
  %v163 = icmp ult i64 %v161, %v162
  br i1 %v163, label %bb29, label %bb70
bb28:
  call void @llvm.nvvm.barrier0() #0
  br label %bb31
bb29:
  %v165 = extractvalue { ptr, i64 } %v25, 0
  %v166 = getelementptr inbounds i16, ptr %v165, i64 %v161
  %v167 = load i16, ptr %v166, align 2
  %v168 = zext i16 %v167 to i32
  %v169 = and i32 16, 31
  %v170 = shl i32 %v168, %v169
  %v171 = bitcast i32 %v170 to float
  %v172 = extractvalue { ptr, i64 } %v28, 1
  %v173 = icmp ult i64 %v159, %v172
  br i1 %v173, label %bb30, label %bb71
bb30:
  %v174 = extractvalue { ptr, i64 } %v28, 0
  %v175 = getelementptr inbounds i16, ptr %v174, i64 %v159
  %v176 = load i16, ptr %v175, align 2
  %v177 = zext i16 %v176 to i32
  %v178 = and i32 16, 31
  %v179 = shl i32 %v177, %v178
  %v180 = bitcast i32 %v179 to float
  %v181 = fmul contract float %v180, %v140
  %v182 = fadd contract float %v171, %v181
  %v183 = bitcast float %v182 to i32
  %v184 = and i32 16, 31
  %v185 = lshr i32 %v183, %v184
  %v186 = trunc i32 %v185 to i16
  %v187 = extractvalue { ptr, i64 } %v25, 0
  %v188 = getelementptr inbounds i16, ptr %v187, i64 %v161
  store i16 %v186, ptr %v188, align 2
  br label %bb26
bb31:
  %v189 = extractvalue { i64, i64 } %v71, 0
  %v190 = extractvalue { i64, i64 } %v71, 1
  %v191 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v189, i64 %v190, i64 %v41) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v191, ptr %v34, align 8
  br label %bb60
bb32:
  %v192 = phi float [ %v230, %bb36 ], [ 0.0, %bb60 ]
  %v193 = phi i64 [ %v320, %bb36 ], [ %v306, %bb60 ]
  %v194 = phi i64 [ %v321, %bb36 ], [ %v309, %bb60 ]
  %v195 = add i64 %v311, 1
  %v196 = icmp eq i64 %v195, 0
  %v197 = select i1 %v196, i8 0, i8 1
  %v198 = insertvalue { i8, { { i64 } } } undef, i8 %v197, 0
  %v199 = insertvalue { i8, { { i64 } } } %v198, i64 %v195, 1, 0, 0
  %v200 = extractvalue { i8, { { i64 } } } %v199, 0
  %v201 = zext i8 %v200 to i64
  %v202 = icmp eq i64 %v201, 1
  %v203 = extractvalue { i8, { { i64 } } } %v199, 1
  %v204 = alloca { { i64 } }, align 8
  store { { i64 } } %v203, ptr %v204, align 8
  %v205 = load i64, ptr %v204, align 8
  %v206 = icmp ugt i64 %v194, 0
  %v207 = xor i1 %v206, 1
  br i1 %v207, label %bb62, label %bb61
bb33:
  %v208 = extractvalue { i64, i64 } %v319, 1
  %v209 = mul i64 %v37, %v42
  %v210 = add i64 %v209, %v208
  %v211 = extractvalue { ptr, i64 } %v25, 1
  %v212 = icmp ult i64 %v210, %v211
  br i1 %v212, label %bb35, label %bb72
bb34:
  store float %v192, ptr %v97, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb37
bb35:
  %v214 = extractvalue { ptr, i64 } %v25, 0
  %v215 = getelementptr inbounds i16, ptr %v214, i64 %v210
  %v216 = load i16, ptr %v215, align 2
  %v217 = zext i16 %v216 to i32
  %v218 = and i32 16, 31
  %v219 = shl i32 %v217, %v218
  %v220 = bitcast i32 %v219 to float
  %v221 = icmp ult i64 %v208, %v61
  br i1 %v221, label %bb36, label %bb73
bb36:
  %v222 = extractvalue { ptr, i64 } %v27, 0
  %v223 = getelementptr inbounds i16, ptr %v222, i64 %v208
  %v224 = load i16, ptr %v223, align 2
  %v225 = zext i16 %v224 to i32
  %v226 = and i32 16, 31
  %v227 = shl i32 %v225, %v226
  %v228 = bitcast i32 %v227 to float
  %v229 = fmul contract float %v220, %v228
  %v230 = fadd contract float %v192, %v229
  br label %bb32
bb37:
  %v231 = udiv i64 %v41, 2
  br label %bb38
bb38:
  %v232 = phi i64 [ %v231, %bb37 ], [ %v249, %bb45 ]
  %v233 = icmp ugt i64 %v232, 0
  %v234 = xor i1 %v233, 1
  br i1 %v234, label %bb46, label %bb39
bb39:
  call void @llvm.nvvm.barrier0() #0
  br label %bb40
bb40:
  %v236 = icmp ult i64 %v39, %v232
  %v237 = xor i1 %v236, 1
  br i1 %v237, label %bb44, label %bb41
bb41:
  %v238 = add i64 %v39, %v232
  %v239 = icmp ult i64 %v238, %v41
  %v240 = xor i1 %v239, 1
  br i1 %v240, label %bb43, label %bb42
bb42:
  %v241 = add i64 %v39, %v232
  %v242 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_gdn_update_bf16, i64 %v241
  %v243 = addrspacecast ptr addrspace(3) %v242 to ptr
  %v244 = load float, ptr %v243, align 4
  %v245 = load float, ptr %v97, align 4
  %v246 = fadd contract float %v245, %v244
  store float %v246, ptr %v97, align 4
  br label %bb45
bb43:
  br label %bb45
bb44:
  br label %bb45
bb45:
  %v247 = zext i32 1 to i64
  %v248 = and i64 %v247, 63
  %v249 = lshr i64 %v232, %v248
  br label %bb38
bb46:
  call void @llvm.nvvm.barrier0() #0
  br label %bb47
bb47:
  %v251 = icmp eq i64 %v39, 0
  br i1 %v251, label %bb48, label %bb49
bb48:
  %v252 = addrspacecast ptr addrspace(3) @__dynamic_smem_infers_gdn_update_bf16 to ptr
  %v253 = load float, ptr %v252, align 4
  %v254 = bitcast float %v253 to i32
  %v255 = and i32 16, 31
  %v256 = lshr i32 %v254, %v255
  %v257 = trunc i32 %v256 to i16
  %v258 = extractvalue { ptr, i64 } %v26, 0
  %v259 = getelementptr inbounds i16, ptr %v258, i64 %v37
  store i16 %v257, ptr %v259, align 2
  br label %bb49
bb49:
  ret void
bb50:
  %v260 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 0
  %v261 = getelementptr inbounds { i64, i64 }, ptr %v260, i32 0, i32 0
  %v262 = load i64, ptr %v261, align 8
  %v263 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 0
  %v264 = getelementptr inbounds { i64, i64 }, ptr %v263, i32 0, i32 1
  %v265 = load i64, ptr %v264, align 8
  %v266 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 1
  %v267 = load i64, ptr %v266, align 8
  %v268 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 2
  %v269 = load i1, ptr %v268, align 1
  br label %bb9
bb51:
  %v270 = add i64 %v76, %v88
  %v271 = sub i64 %v77, 1
  %v272 = insertvalue { i64, i64 } undef, i64 1, 0
  %v273 = insertvalue { i64, i64 } %v272, i64 %v76, 1
  br label %bb53
bb52:
  %v274 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb53
bb53:
  %v275 = phi { i64, i64 } [ %v273, %bb51 ], [ %v274, %bb52 ]
  %v276 = phi i64 [ %v270, %bb51 ], [ %v76, %bb52 ]
  %v277 = phi i64 [ %v271, %bb51 ], [ %v77, %bb52 ]
  %v278 = extractvalue { i64, i64 } %v275, 0
  %v279 = bitcast i64 %v278 to i64
  %v280 = icmp eq i64 %v279, 0
  br i1 %v280, label %bb12, label %bb54
bb54:
  %v281 = icmp eq i64 %v279, 1
  br i1 %v281, label %bb11, label %bb10
bb55:
  %v282 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 0
  %v283 = getelementptr inbounds { i64, i64 }, ptr %v282, i32 0, i32 0
  %v284 = load i64, ptr %v283, align 8
  %v285 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 0
  %v286 = getelementptr inbounds { i64, i64 }, ptr %v285, i32 0, i32 1
  %v287 = load i64, ptr %v286, align 8
  %v288 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 1
  %v289 = load i64, ptr %v288, align 8
  %v290 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 2
  %v291 = load i1, ptr %v290, align 1
  br label %bb26
bb56:
  %v292 = add i64 %v144, %v156
  %v293 = sub i64 %v145, 1
  %v294 = insertvalue { i64, i64 } undef, i64 1, 0
  %v295 = insertvalue { i64, i64 } %v294, i64 %v144, 1
  br label %bb58
bb57:
  %v296 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb58
bb58:
  %v297 = phi { i64, i64 } [ %v295, %bb56 ], [ %v296, %bb57 ]
  %v298 = phi i64 [ %v292, %bb56 ], [ %v144, %bb57 ]
  %v299 = phi i64 [ %v293, %bb56 ], [ %v145, %bb57 ]
  %v300 = extractvalue { i64, i64 } %v297, 0
  %v301 = bitcast i64 %v300 to i64
  %v302 = icmp eq i64 %v301, 0
  br i1 %v302, label %bb28, label %bb59
bb59:
  %v303 = icmp eq i64 %v301, 1
  br i1 %v303, label %bb27, label %bb10
bb60:
  %v304 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v305 = getelementptr inbounds { i64, i64 }, ptr %v304, i32 0, i32 0
  %v306 = load i64, ptr %v305, align 8
  %v307 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v308 = getelementptr inbounds { i64, i64 }, ptr %v307, i32 0, i32 1
  %v309 = load i64, ptr %v308, align 8
  %v310 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 1
  %v311 = load i64, ptr %v310, align 8
  %v312 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 2
  %v313 = load i1, ptr %v312, align 1
  br label %bb32
bb61:
  %v314 = add i64 %v193, %v205
  %v315 = sub i64 %v194, 1
  %v316 = insertvalue { i64, i64 } undef, i64 1, 0
  %v317 = insertvalue { i64, i64 } %v316, i64 %v193, 1
  br label %bb63
bb62:
  %v318 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb63
bb63:
  %v319 = phi { i64, i64 } [ %v317, %bb61 ], [ %v318, %bb62 ]
  %v320 = phi i64 [ %v314, %bb61 ], [ %v193, %bb62 ]
  %v321 = phi i64 [ %v315, %bb61 ], [ %v194, %bb62 ]
  %v322 = extractvalue { i64, i64 } %v319, 0
  %v323 = bitcast i64 %v322 to i64
  %v324 = icmp eq i64 %v323, 0
  br i1 %v324, label %bb34, label %bb64
bb64:
  %v325 = icmp eq i64 %v323, 1
  br i1 %v325, label %bb33, label %bb10
bb65:
  unreachable
bb66:
  unreachable
bb67:
  unreachable
bb68:
  unreachable
bb69:
  unreachable
bb70:
  unreachable
bb71:
  unreachable
bb72:
  unreachable
bb73:
  unreachable
}

define void @infers_gdn_recurrent_step_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, ptr %v12, i64 %v13, ptr %v14, i64 %v15, ptr %v16, i64 %v17, i32 %v18, i32 %v19, i32 %v20) #0 {
entry:
  %v21 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v1, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v3, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v5, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v7, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v9, 1
  %v31 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v32 = insertvalue { ptr, i64 } %v31, i64 %v11, 1
  %v33 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v34 = insertvalue { ptr, i64 } %v33, i64 %v13, 1
  %v35 = insertvalue { ptr, i64 } undef, ptr %v14, 0
  %v36 = insertvalue { ptr, i64 } %v35, i64 %v15, 1
  %v37 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v38 = insertvalue { ptr, i64 } %v37, i64 %v17, 1
  br label %bb0
bb0:
  %v39 = phi { ptr, i64 } [ %v22, %entry ]
  %v40 = phi { ptr, i64 } [ %v24, %entry ]
  %v41 = phi { ptr, i64 } [ %v26, %entry ]
  %v42 = phi { ptr, i64 } [ %v28, %entry ]
  %v43 = phi { ptr, i64 } [ %v30, %entry ]
  %v44 = phi { ptr, i64 } [ %v32, %entry ]
  %v45 = phi { ptr, i64 } [ %v34, %entry ]
  %v46 = phi { ptr, i64 } [ %v36, %entry ]
  %v47 = phi { ptr, i64 } [ %v38, %entry ]
  %v48 = phi i32 [ %v18, %entry ]
  %v49 = phi i32 [ %v19, %entry ]
  %v50 = phi i32 [ %v20, %entry ]
  %v51 = alloca {  }, align 1
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v53 = mul i32 %v48, %v50
  %v54 = zext i32 %v53 to i64
  %v55 = bitcast ptr %v51 to ptr
  %v56 = call i64 @cuda_device____internal__index_1d(ptr %v55) #0
  br label %bb2
bb2:
  %v57 = icmp uge i64 %v56, %v54
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb4, label %bb3
bb3:
  br label %bb47
bb4:
  %v59 = zext i32 %v50 to i64
  %v60 = icmp eq i64 %v59, 0
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb5, label %bb71
bb5:
  %v62 = udiv i64 %v56, %v59
  %v63 = urem i64 %v56, %v59
  %v64 = zext i32 %v49 to i64
  %v65 = uitofp i64 %v64 to float
  %v66 = call float @__nv_sqrtf(float %v65) #0
  br label %bb48
bb6:
  %v67 = extractvalue { ptr, i64 } %v44, 0
  %v68 = getelementptr inbounds float, ptr %v67, i64 %v62
  %v69 = load float, ptr %v68, align 4
  %v70 = call float @__nv_expf(float %v69) #0
  br label %bb7
bb7:
  %v71 = extractvalue { ptr, i64 } %v42, 1
  %v72 = icmp ult i64 %v62, %v71
  br i1 %v72, label %bb8, label %bb72
bb8:
  %v73 = extractvalue { ptr, i64 } %v42, 0
  %v74 = getelementptr inbounds i16, ptr %v73, i64 %v62
  %v75 = load i16, ptr %v74, align 2
  %v76 = zext i16 %v75 to i32
  %v77 = and i32 16, 31
  %v78 = shl i32 %v76, %v77
  %v79 = bitcast i32 %v78 to float
  %v80 = extractvalue { ptr, i64 } %v45, 1
  %v81 = icmp ult i64 %v62, %v80
  br i1 %v81, label %bb9, label %bb73
bb9:
  %v82 = extractvalue { ptr, i64 } %v45, 0
  %v83 = getelementptr inbounds float, ptr %v82, i64 %v62
  %v84 = load float, ptr %v83, align 4
  %v85 = fadd contract float %v79, %v84
  %v86 = fcmp ogt float %v85, 20.0
  %v87 = xor i1 %v86, 1
  br i1 %v87, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v88 = fcmp olt float %v85, -20.0
  %v89 = xor i1 %v88, 1
  br i1 %v89, label %bb13, label %bb12
bb12:
  br label %bb16
bb13:
  %v90 = call float @__nv_expf(float %v85) #0
  br label %bb14
bb14:
  %v91 = fadd contract float 1.0, %v90
  %v92 = call float @__nv_logf(float %v91) #0
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v93 = phi float [ 0.0, %bb12 ], [ %v92, %bb15 ]
  br label %bb17
bb17:
  %v94 = phi float [ %v85, %bb10 ], [ %v93, %bb16 ]
  %v95 = fneg float %v70
  %v96 = fmul contract float %v95, %v94
  %v97 = call float @__nv_expf(float %v96) #0
  br label %bb18
bb18:
  %v98 = extractvalue { ptr, i64 } %v43, 1
  %v99 = icmp ult i64 %v62, %v98
  br i1 %v99, label %bb19, label %bb74
bb19:
  %v100 = extractvalue { ptr, i64 } %v43, 0
  %v101 = getelementptr inbounds i16, ptr %v100, i64 %v62
  %v102 = load i16, ptr %v101, align 2
  %v103 = zext i16 %v102 to i32
  %v104 = and i32 16, 31
  %v105 = shl i32 %v103, %v104
  %v106 = bitcast i32 %v105 to float
  %v107 = fneg float %v106
  %v108 = call float @__nv_expf(float %v107) #0
  br label %bb20
bb20:
  %v109 = fadd contract float 1.0, %v108
  %v110 = fdiv contract float 1.0, %v109
  br label %bb21
bb21:
  %v111 = phi float [ 0.0, %bb20 ], [ %v140, %bb26 ]
  %v112 = phi float [ 0.0, %bb20 ], [ %v142, %bb26 ]
  %v113 = phi i64 [ 0, %bb20 ], [ %v260, %bb26 ]
  %v114 = icmp ult i64 %v113, %v64
  %v115 = xor i1 %v114, 1
  br i1 %v115, label %bb50, label %bb49
bb22:
  unreachable
bb23:
  %v116 = extractvalue { i64, i64 } %v259, 1
  %v117 = mul i64 %v62, %v64
  %v118 = add i64 %v117, %v116
  %v119 = extractvalue { ptr, i64 } %v40, 1
  %v120 = icmp ult i64 %v118, %v119
  br i1 %v120, label %bb25, label %bb75
bb24:
  %v121 = fadd contract float %v111, 0.0000009999999974752427
  %v122 = call float @__nv_sqrtf(float %v121) #0
  br label %bb53
bb25:
  %v123 = extractvalue { ptr, i64 } %v40, 0
  %v124 = getelementptr inbounds i16, ptr %v123, i64 %v118
  %v125 = load i16, ptr %v124, align 2
  %v126 = zext i16 %v125 to i32
  %v127 = and i32 16, 31
  %v128 = shl i32 %v126, %v127
  %v129 = bitcast i32 %v128 to float
  %v130 = extractvalue { ptr, i64 } %v39, 1
  %v131 = icmp ult i64 %v118, %v130
  br i1 %v131, label %bb26, label %bb76
bb26:
  %v132 = extractvalue { ptr, i64 } %v39, 0
  %v133 = getelementptr inbounds i16, ptr %v132, i64 %v118
  %v134 = load i16, ptr %v133, align 2
  %v135 = zext i16 %v134 to i32
  %v136 = and i32 16, 31
  %v137 = shl i32 %v135, %v136
  %v138 = bitcast i32 %v137 to float
  %v139 = fmul contract float %v129, %v129
  %v140 = fadd contract float %v111, %v139
  %v141 = fmul contract float %v138, %v138
  %v142 = fadd contract float %v112, %v141
  br label %bb21
bb27:
  %v143 = phi i64 [ %v278, %bb30 ], [ 0, %bb54 ]
  %v144 = icmp ult i64 %v143, %v64
  %v145 = xor i1 %v144, 1
  br i1 %v145, label %bb56, label %bb55
bb28:
  %v146 = extractvalue { i64, i64 } %v277, 1
  %v147 = mul i64 %v146, %v59
  %v148 = add i64 %v272, %v147
  %v149 = extractvalue { ptr, i64 } %v46, 1
  %v150 = icmp ult i64 %v148, %v149
  br i1 %v150, label %bb30, label %bb77
bb29:
  br label %bb31
bb30:
  %v151 = extractvalue { ptr, i64 } %v46, 0
  %v152 = getelementptr inbounds float, ptr %v151, i64 %v148
  %v153 = load float, ptr %v152, align 4
  %v154 = fmul contract float %v153, %v97
  %v155 = extractvalue { ptr, i64 } %v46, 0
  %v156 = getelementptr inbounds float, ptr %v155, i64 %v148
  store float %v154, ptr %v156, align 4
  br label %bb27
bb31:
  %v157 = phi float [ 0.0, %bb29 ], [ %v185, %bb35 ]
  %v158 = phi i64 [ 0, %bb29 ], [ %v288, %bb35 ]
  %v159 = icmp ult i64 %v158, %v64
  %v160 = xor i1 %v159, 1
  br i1 %v160, label %bb60, label %bb59
bb32:
  %v161 = extractvalue { i64, i64 } %v287, 1
  %v162 = mul i64 %v161, %v59
  %v163 = add i64 %v272, %v162
  %v164 = extractvalue { ptr, i64 } %v46, 1
  %v165 = icmp ult i64 %v163, %v164
  br i1 %v165, label %bb34, label %bb78
bb33:
  %v166 = mul i64 %v62, %v59
  %v167 = add i64 %v166, %v63
  %v168 = extractvalue { ptr, i64 } %v41, 1
  %v169 = icmp ult i64 %v167, %v168
  br i1 %v169, label %bb36, label %bb79
bb34:
  %v170 = extractvalue { ptr, i64 } %v46, 0
  %v171 = getelementptr inbounds float, ptr %v170, i64 %v163
  %v172 = load float, ptr %v171, align 4
  %v173 = add i64 %v270, %v161
  %v174 = extractvalue { ptr, i64 } %v40, 1
  %v175 = icmp ult i64 %v173, %v174
  br i1 %v175, label %bb35, label %bb80
bb35:
  %v176 = extractvalue { ptr, i64 } %v40, 0
  %v177 = getelementptr inbounds i16, ptr %v176, i64 %v173
  %v178 = load i16, ptr %v177, align 2
  %v179 = zext i16 %v178 to i32
  %v180 = and i32 16, 31
  %v181 = shl i32 %v179, %v180
  %v182 = bitcast i32 %v181 to float
  %v183 = fmul contract float %v182, %v265
  %v184 = fmul contract float %v172, %v183
  %v185 = fadd contract float %v157, %v184
  br label %bb31
bb36:
  %v186 = extractvalue { ptr, i64 } %v41, 0
  %v187 = getelementptr inbounds i16, ptr %v186, i64 %v167
  %v188 = load i16, ptr %v187, align 2
  %v189 = zext i16 %v188 to i32
  %v190 = and i32 16, 31
  %v191 = shl i32 %v189, %v190
  %v192 = bitcast i32 %v191 to float
  %v193 = fsub contract float %v192, %v157
  %v194 = fmul contract float %v110, %v193
  br label %bb37
bb37:
  %v195 = phi i64 [ 0, %bb36 ], [ %v298, %bb41 ]
  %v196 = icmp ult i64 %v195, %v64
  %v197 = xor i1 %v196, 1
  br i1 %v197, label %bb64, label %bb63
bb38:
  %v198 = extractvalue { i64, i64 } %v297, 1
  %v199 = add i64 %v270, %v198
  %v200 = extractvalue { ptr, i64 } %v40, 1
  %v201 = icmp ult i64 %v199, %v200
  br i1 %v201, label %bb40, label %bb81
bb39:
  br label %bb42
bb40:
  %v202 = extractvalue { ptr, i64 } %v40, 0
  %v203 = getelementptr inbounds i16, ptr %v202, i64 %v199
  %v204 = load i16, ptr %v203, align 2
  %v205 = zext i16 %v204 to i32
  %v206 = and i32 16, 31
  %v207 = shl i32 %v205, %v206
  %v208 = bitcast i32 %v207 to float
  %v209 = fmul contract float %v208, %v265
  %v210 = fmul contract float %v209, %v194
  %v211 = mul i64 %v198, %v59
  %v212 = add i64 %v272, %v211
  %v213 = extractvalue { ptr, i64 } %v46, 1
  %v214 = icmp ult i64 %v212, %v213
  br i1 %v214, label %bb41, label %bb82
bb41:
  %v215 = extractvalue { ptr, i64 } %v46, 0
  %v216 = getelementptr inbounds float, ptr %v215, i64 %v212
  %v217 = load float, ptr %v216, align 4
  %v218 = fadd contract float %v217, %v210
  %v219 = extractvalue { ptr, i64 } %v46, 0
  %v220 = getelementptr inbounds float, ptr %v219, i64 %v212
  store float %v218, ptr %v220, align 4
  br label %bb37
bb42:
  %v221 = phi float [ 0.0, %bb39 ], [ %v251, %bb46 ]
  %v222 = phi i64 [ 0, %bb39 ], [ %v308, %bb46 ]
  %v223 = icmp ult i64 %v222, %v64
  %v224 = xor i1 %v223, 1
  br i1 %v224, label %bb68, label %bb67
bb43:
  %v225 = extractvalue { i64, i64 } %v307, 1
  %v226 = mul i64 %v225, %v59
  %v227 = add i64 %v272, %v226
  %v228 = extractvalue { ptr, i64 } %v46, 1
  %v229 = icmp ult i64 %v227, %v228
  br i1 %v229, label %bb45, label %bb83
bb44:
  %v230 = bitcast float %v221 to i32
  %v231 = and i32 16, 31
  %v232 = lshr i32 %v230, %v231
  %v233 = trunc i32 %v232 to i16
  %v234 = extractvalue { ptr, i64 } %v47, 0
  %v235 = getelementptr inbounds i16, ptr %v234, i64 %v167
  store i16 %v233, ptr %v235, align 2
  br label %bb47
bb45:
  %v236 = extractvalue { ptr, i64 } %v46, 0
  %v237 = getelementptr inbounds float, ptr %v236, i64 %v227
  %v238 = load float, ptr %v237, align 4
  %v239 = add i64 %v270, %v225
  %v240 = extractvalue { ptr, i64 } %v39, 1
  %v241 = icmp ult i64 %v239, %v240
  br i1 %v241, label %bb46, label %bb84
bb46:
  %v242 = extractvalue { ptr, i64 } %v39, 0
  %v243 = getelementptr inbounds i16, ptr %v242, i64 %v239
  %v244 = load i16, ptr %v243, align 2
  %v245 = zext i16 %v244 to i32
  %v246 = and i32 16, 31
  %v247 = shl i32 %v245, %v246
  %v248 = bitcast i32 %v247 to float
  %v249 = fmul contract float %v248, %v269
  %v250 = fmul contract float %v238, %v249
  %v251 = fadd contract float %v221, %v250
  br label %bb42
bb47:
  ret void
bb48:
  %v252 = fdiv contract float 1.0, %v66
  %v253 = extractvalue { ptr, i64 } %v44, 1
  %v254 = icmp ult i64 %v62, %v253
  br i1 %v254, label %bb6, label %bb85
bb49:
  %v255 = add i64 %v113, 1
  %v256 = insertvalue { i64, i64 } undef, i64 1, 0
  %v257 = insertvalue { i64, i64 } %v256, i64 %v113, 1
  br label %bb51
bb50:
  %v258 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb51
bb51:
  %v259 = phi { i64, i64 } [ %v257, %bb49 ], [ %v258, %bb50 ]
  %v260 = phi i64 [ %v255, %bb49 ], [ %v113, %bb50 ]
  %v261 = extractvalue { i64, i64 } %v259, 0
  %v262 = bitcast i64 %v261 to i64
  %v263 = icmp eq i64 %v262, 0
  br i1 %v263, label %bb24, label %bb52
bb52:
  %v264 = icmp eq i64 %v262, 1
  br i1 %v264, label %bb23, label %bb22
bb53:
  %v265 = fdiv contract float 1.0, %v122
  %v266 = fadd contract float %v112, 0.0000009999999974752427
  %v267 = call float @__nv_sqrtf(float %v266) #0
  br label %bb54
bb54:
  %v268 = fdiv contract float 1.0, %v267
  %v269 = fmul contract float %v268, %v252
  %v270 = mul i64 %v62, %v64
  %v271 = mul i64 %v270, %v59
  %v272 = add i64 %v271, %v63
  br label %bb27
bb55:
  %v273 = add i64 %v143, 1
  %v274 = insertvalue { i64, i64 } undef, i64 1, 0
  %v275 = insertvalue { i64, i64 } %v274, i64 %v143, 1
  br label %bb57
bb56:
  %v276 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb57
bb57:
  %v277 = phi { i64, i64 } [ %v275, %bb55 ], [ %v276, %bb56 ]
  %v278 = phi i64 [ %v273, %bb55 ], [ %v143, %bb56 ]
  %v279 = extractvalue { i64, i64 } %v277, 0
  %v280 = bitcast i64 %v279 to i64
  %v281 = icmp eq i64 %v280, 0
  br i1 %v281, label %bb29, label %bb58
bb58:
  %v282 = icmp eq i64 %v280, 1
  br i1 %v282, label %bb28, label %bb22
bb59:
  %v283 = add i64 %v158, 1
  %v284 = insertvalue { i64, i64 } undef, i64 1, 0
  %v285 = insertvalue { i64, i64 } %v284, i64 %v158, 1
  br label %bb61
bb60:
  %v286 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb61
bb61:
  %v287 = phi { i64, i64 } [ %v285, %bb59 ], [ %v286, %bb60 ]
  %v288 = phi i64 [ %v283, %bb59 ], [ %v158, %bb60 ]
  %v289 = extractvalue { i64, i64 } %v287, 0
  %v290 = bitcast i64 %v289 to i64
  %v291 = icmp eq i64 %v290, 0
  br i1 %v291, label %bb33, label %bb62
bb62:
  %v292 = icmp eq i64 %v290, 1
  br i1 %v292, label %bb32, label %bb22
bb63:
  %v293 = add i64 %v195, 1
  %v294 = insertvalue { i64, i64 } undef, i64 1, 0
  %v295 = insertvalue { i64, i64 } %v294, i64 %v195, 1
  br label %bb65
bb64:
  %v296 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb65
bb65:
  %v297 = phi { i64, i64 } [ %v295, %bb63 ], [ %v296, %bb64 ]
  %v298 = phi i64 [ %v293, %bb63 ], [ %v195, %bb64 ]
  %v299 = extractvalue { i64, i64 } %v297, 0
  %v300 = bitcast i64 %v299 to i64
  %v301 = icmp eq i64 %v300, 0
  br i1 %v301, label %bb39, label %bb66
bb66:
  %v302 = icmp eq i64 %v300, 1
  br i1 %v302, label %bb38, label %bb22
bb67:
  %v303 = add i64 %v222, 1
  %v304 = insertvalue { i64, i64 } undef, i64 1, 0
  %v305 = insertvalue { i64, i64 } %v304, i64 %v222, 1
  br label %bb69
bb68:
  %v306 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb69
bb69:
  %v307 = phi { i64, i64 } [ %v305, %bb67 ], [ %v306, %bb68 ]
  %v308 = phi i64 [ %v303, %bb67 ], [ %v222, %bb68 ]
  %v309 = extractvalue { i64, i64 } %v307, 0
  %v310 = bitcast i64 %v309 to i64
  %v311 = icmp eq i64 %v310, 0
  br i1 %v311, label %bb44, label %bb70
bb70:
  %v312 = icmp eq i64 %v310, 1
  br i1 %v312, label %bb43, label %bb22
bb71:
  unreachable
bb72:
  unreachable
bb73:
  unreachable
bb74:
  unreachable
bb75:
  unreachable
bb76:
  unreachable
bb77:
  unreachable
bb78:
  unreachable
bb79:
  unreachable
bb80:
  unreachable
bb81:
  unreachable
bb82:
  unreachable
bb83:
  unreachable
bb84:
  unreachable
bb85:
  unreachable
}

define void @infers_gdn_gated_delta_update_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, ptr %v10, i64 %v11, ptr %v12, i64 %v13, ptr %v14, i64 %v15, ptr %v16, i64 %v17, i32 %v18, i32 %v19, i32 %v20) #0 {
entry:
  %v21 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v1, 1
  %v23 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v24 = insertvalue { ptr, i64 } %v23, i64 %v3, 1
  %v25 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v26 = insertvalue { ptr, i64 } %v25, i64 %v5, 1
  %v27 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v28 = insertvalue { ptr, i64 } %v27, i64 %v7, 1
  %v29 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v30 = insertvalue { ptr, i64 } %v29, i64 %v9, 1
  %v31 = insertvalue { ptr, i64 } undef, ptr %v10, 0
  %v32 = insertvalue { ptr, i64 } %v31, i64 %v11, 1
  %v33 = insertvalue { ptr, i64 } undef, ptr %v12, 0
  %v34 = insertvalue { ptr, i64 } %v33, i64 %v13, 1
  %v35 = insertvalue { ptr, i64 } undef, ptr %v14, 0
  %v36 = insertvalue { ptr, i64 } %v35, i64 %v15, 1
  %v37 = insertvalue { ptr, i64 } undef, ptr %v16, 0
  %v38 = insertvalue { ptr, i64 } %v37, i64 %v17, 1
  br label %bb0
bb0:
  %v39 = phi { ptr, i64 } [ %v22, %entry ]
  %v40 = phi { ptr, i64 } [ %v24, %entry ]
  %v41 = phi { ptr, i64 } [ %v26, %entry ]
  %v42 = phi { ptr, i64 } [ %v28, %entry ]
  %v43 = phi { ptr, i64 } [ %v30, %entry ]
  %v44 = phi { ptr, i64 } [ %v32, %entry ]
  %v45 = phi { ptr, i64 } [ %v34, %entry ]
  %v46 = phi { ptr, i64 } [ %v36, %entry ]
  %v47 = phi { ptr, i64 } [ %v38, %entry ]
  %v48 = phi i32 [ %v18, %entry ]
  %v49 = phi i32 [ %v19, %entry ]
  %v50 = phi i32 [ %v20, %entry ]
  %v51 = alloca {  }, align 1
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v53 = mul i32 %v48, %v50
  %v54 = zext i32 %v53 to i64
  %v55 = bitcast ptr %v51 to ptr
  %v56 = call i64 @cuda_device____internal__index_1d(ptr %v55) #0
  br label %bb2
bb2:
  %v57 = icmp uge i64 %v56, %v54
  %v58 = xor i1 %v57, 1
  br i1 %v58, label %bb4, label %bb3
bb3:
  br label %bb47
bb4:
  %v59 = zext i32 %v50 to i64
  %v60 = icmp eq i64 %v59, 0
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb5, label %bb71
bb5:
  %v62 = udiv i64 %v56, %v59
  %v63 = urem i64 %v56, %v59
  %v64 = zext i32 %v49 to i64
  %v65 = uitofp i64 %v64 to float
  %v66 = call float @__nv_sqrtf(float %v65) #0
  br label %bb48
bb6:
  %v67 = extractvalue { ptr, i64 } %v44, 0
  %v68 = getelementptr inbounds float, ptr %v67, i64 %v62
  %v69 = load float, ptr %v68, align 4
  %v70 = call float @__nv_expf(float %v69) #0
  br label %bb7
bb7:
  %v71 = extractvalue { ptr, i64 } %v42, 1
  %v72 = icmp ult i64 %v62, %v71
  br i1 %v72, label %bb8, label %bb72
bb8:
  %v73 = extractvalue { ptr, i64 } %v42, 0
  %v74 = getelementptr inbounds i16, ptr %v73, i64 %v62
  %v75 = load i16, ptr %v74, align 2
  %v76 = zext i16 %v75 to i32
  %v77 = and i32 16, 31
  %v78 = shl i32 %v76, %v77
  %v79 = bitcast i32 %v78 to float
  %v80 = extractvalue { ptr, i64 } %v45, 1
  %v81 = icmp ult i64 %v62, %v80
  br i1 %v81, label %bb9, label %bb73
bb9:
  %v82 = extractvalue { ptr, i64 } %v45, 0
  %v83 = getelementptr inbounds float, ptr %v82, i64 %v62
  %v84 = load float, ptr %v83, align 4
  %v85 = fadd contract float %v79, %v84
  %v86 = fcmp ogt float %v85, 20.0
  %v87 = xor i1 %v86, 1
  br i1 %v87, label %bb11, label %bb10
bb10:
  br label %bb17
bb11:
  %v88 = fcmp olt float %v85, -20.0
  %v89 = xor i1 %v88, 1
  br i1 %v89, label %bb13, label %bb12
bb12:
  br label %bb16
bb13:
  %v90 = call float @__nv_expf(float %v85) #0
  br label %bb14
bb14:
  %v91 = fadd contract float 1.0, %v90
  %v92 = call float @__nv_logf(float %v91) #0
  br label %bb15
bb15:
  br label %bb16
bb16:
  %v93 = phi float [ 0.0, %bb12 ], [ %v92, %bb15 ]
  br label %bb17
bb17:
  %v94 = phi float [ %v85, %bb10 ], [ %v93, %bb16 ]
  %v95 = fneg float %v70
  %v96 = fmul contract float %v95, %v94
  %v97 = extractvalue { ptr, i64 } %v43, 1
  %v98 = icmp ult i64 %v62, %v97
  br i1 %v98, label %bb18, label %bb74
bb18:
  %v99 = extractvalue { ptr, i64 } %v43, 0
  %v100 = getelementptr inbounds i16, ptr %v99, i64 %v62
  %v101 = load i16, ptr %v100, align 2
  %v102 = zext i16 %v101 to i32
  %v103 = and i32 16, 31
  %v104 = shl i32 %v102, %v103
  %v105 = bitcast i32 %v104 to float
  %v106 = fneg float %v105
  %v107 = call float @__nv_expf(float %v106) #0
  br label %bb19
bb19:
  %v108 = fadd contract float 1.0, %v107
  %v109 = fdiv contract float 1.0, %v108
  %v110 = call float @__nv_expf(float %v96) #0
  br label %bb20
bb20:
  br label %bb21
bb21:
  %v111 = phi float [ 0.0, %bb20 ], [ %v140, %bb26 ]
  %v112 = phi float [ 0.0, %bb20 ], [ %v142, %bb26 ]
  %v113 = phi i64 [ 0, %bb20 ], [ %v261, %bb26 ]
  %v114 = icmp ult i64 %v113, %v64
  %v115 = xor i1 %v114, 1
  br i1 %v115, label %bb50, label %bb49
bb22:
  unreachable
bb23:
  %v116 = extractvalue { i64, i64 } %v260, 1
  %v117 = mul i64 %v62, %v64
  %v118 = add i64 %v117, %v116
  %v119 = extractvalue { ptr, i64 } %v40, 1
  %v120 = icmp ult i64 %v118, %v119
  br i1 %v120, label %bb25, label %bb75
bb24:
  %v121 = fadd contract float %v111, 0.0000009999999974752427
  %v122 = call float @__nv_sqrtf(float %v121) #0
  br label %bb53
bb25:
  %v123 = extractvalue { ptr, i64 } %v40, 0
  %v124 = getelementptr inbounds i16, ptr %v123, i64 %v118
  %v125 = load i16, ptr %v124, align 2
  %v126 = zext i16 %v125 to i32
  %v127 = and i32 16, 31
  %v128 = shl i32 %v126, %v127
  %v129 = bitcast i32 %v128 to float
  %v130 = extractvalue { ptr, i64 } %v39, 1
  %v131 = icmp ult i64 %v118, %v130
  br i1 %v131, label %bb26, label %bb76
bb26:
  %v132 = extractvalue { ptr, i64 } %v39, 0
  %v133 = getelementptr inbounds i16, ptr %v132, i64 %v118
  %v134 = load i16, ptr %v133, align 2
  %v135 = zext i16 %v134 to i32
  %v136 = and i32 16, 31
  %v137 = shl i32 %v135, %v136
  %v138 = bitcast i32 %v137 to float
  %v139 = fmul contract float %v129, %v129
  %v140 = fadd contract float %v111, %v139
  %v141 = fmul contract float %v138, %v138
  %v142 = fadd contract float %v112, %v141
  br label %bb21
bb27:
  %v143 = phi i64 [ %v278, %bb30 ], [ 0, %bb54 ]
  %v144 = icmp ult i64 %v143, %v64
  %v145 = xor i1 %v144, 1
  br i1 %v145, label %bb56, label %bb55
bb28:
  %v146 = extractvalue { i64, i64 } %v277, 1
  %v147 = mul i64 %v146, %v59
  %v148 = add i64 %v272, %v147
  %v149 = extractvalue { ptr, i64 } %v46, 1
  %v150 = icmp ult i64 %v148, %v149
  br i1 %v150, label %bb30, label %bb77
bb29:
  br label %bb31
bb30:
  %v151 = extractvalue { ptr, i64 } %v46, 0
  %v152 = getelementptr inbounds float, ptr %v151, i64 %v148
  %v153 = load float, ptr %v152, align 4
  %v154 = fmul contract float %v153, %v110
  %v155 = extractvalue { ptr, i64 } %v46, 0
  %v156 = getelementptr inbounds float, ptr %v155, i64 %v148
  store float %v154, ptr %v156, align 4
  br label %bb27
bb31:
  %v157 = phi float [ 0.0, %bb29 ], [ %v185, %bb35 ]
  %v158 = phi i64 [ 0, %bb29 ], [ %v288, %bb35 ]
  %v159 = icmp ult i64 %v158, %v64
  %v160 = xor i1 %v159, 1
  br i1 %v160, label %bb60, label %bb59
bb32:
  %v161 = extractvalue { i64, i64 } %v287, 1
  %v162 = mul i64 %v161, %v59
  %v163 = add i64 %v272, %v162
  %v164 = extractvalue { ptr, i64 } %v46, 1
  %v165 = icmp ult i64 %v163, %v164
  br i1 %v165, label %bb34, label %bb78
bb33:
  %v166 = mul i64 %v62, %v59
  %v167 = add i64 %v166, %v63
  %v168 = extractvalue { ptr, i64 } %v41, 1
  %v169 = icmp ult i64 %v167, %v168
  br i1 %v169, label %bb36, label %bb79
bb34:
  %v170 = extractvalue { ptr, i64 } %v46, 0
  %v171 = getelementptr inbounds float, ptr %v170, i64 %v163
  %v172 = load float, ptr %v171, align 4
  %v173 = add i64 %v270, %v161
  %v174 = extractvalue { ptr, i64 } %v40, 1
  %v175 = icmp ult i64 %v173, %v174
  br i1 %v175, label %bb35, label %bb80
bb35:
  %v176 = extractvalue { ptr, i64 } %v40, 0
  %v177 = getelementptr inbounds i16, ptr %v176, i64 %v173
  %v178 = load i16, ptr %v177, align 2
  %v179 = zext i16 %v178 to i32
  %v180 = and i32 16, 31
  %v181 = shl i32 %v179, %v180
  %v182 = bitcast i32 %v181 to float
  %v183 = fmul contract float %v182, %v266
  %v184 = fmul contract float %v172, %v183
  %v185 = fadd contract float %v157, %v184
  br label %bb31
bb36:
  %v186 = extractvalue { ptr, i64 } %v41, 0
  %v187 = getelementptr inbounds i16, ptr %v186, i64 %v167
  %v188 = load i16, ptr %v187, align 2
  %v189 = zext i16 %v188 to i32
  %v190 = and i32 16, 31
  %v191 = shl i32 %v189, %v190
  %v192 = bitcast i32 %v191 to float
  %v193 = fsub contract float %v192, %v157
  %v194 = fmul contract float %v109, %v193
  br label %bb37
bb37:
  %v195 = phi i64 [ 0, %bb36 ], [ %v298, %bb41 ]
  %v196 = icmp ult i64 %v195, %v64
  %v197 = xor i1 %v196, 1
  br i1 %v197, label %bb64, label %bb63
bb38:
  %v198 = extractvalue { i64, i64 } %v297, 1
  %v199 = add i64 %v270, %v198
  %v200 = extractvalue { ptr, i64 } %v40, 1
  %v201 = icmp ult i64 %v199, %v200
  br i1 %v201, label %bb40, label %bb81
bb39:
  br label %bb42
bb40:
  %v202 = extractvalue { ptr, i64 } %v40, 0
  %v203 = getelementptr inbounds i16, ptr %v202, i64 %v199
  %v204 = load i16, ptr %v203, align 2
  %v205 = zext i16 %v204 to i32
  %v206 = and i32 16, 31
  %v207 = shl i32 %v205, %v206
  %v208 = bitcast i32 %v207 to float
  %v209 = fmul contract float %v208, %v266
  %v210 = fmul contract float %v209, %v194
  %v211 = mul i64 %v198, %v59
  %v212 = add i64 %v272, %v211
  %v213 = extractvalue { ptr, i64 } %v46, 1
  %v214 = icmp ult i64 %v212, %v213
  br i1 %v214, label %bb41, label %bb82
bb41:
  %v215 = extractvalue { ptr, i64 } %v46, 0
  %v216 = getelementptr inbounds float, ptr %v215, i64 %v212
  %v217 = load float, ptr %v216, align 4
  %v218 = fadd contract float %v217, %v210
  %v219 = extractvalue { ptr, i64 } %v46, 0
  %v220 = getelementptr inbounds float, ptr %v219, i64 %v212
  store float %v218, ptr %v220, align 4
  br label %bb37
bb42:
  %v221 = phi float [ 0.0, %bb39 ], [ %v252, %bb46 ]
  %v222 = phi i64 [ 0, %bb39 ], [ %v308, %bb46 ]
  %v223 = icmp ult i64 %v222, %v64
  %v224 = xor i1 %v223, 1
  br i1 %v224, label %bb68, label %bb67
bb43:
  %v225 = extractvalue { i64, i64 } %v307, 1
  %v226 = mul i64 %v225, %v59
  %v227 = add i64 %v272, %v226
  %v228 = extractvalue { ptr, i64 } %v46, 1
  %v229 = icmp ult i64 %v227, %v228
  br i1 %v229, label %bb45, label %bb83
bb44:
  %v230 = bitcast float %v221 to i32
  %v231 = and i32 16, 31
  %v232 = lshr i32 %v230, %v231
  %v233 = trunc i32 %v232 to i16
  %v234 = extractvalue { ptr, i64 } %v47, 0
  %v235 = getelementptr inbounds i16, ptr %v234, i64 %v167
  store i16 %v233, ptr %v235, align 2
  br label %bb47
bb45:
  %v236 = extractvalue { ptr, i64 } %v46, 0
  %v237 = getelementptr inbounds float, ptr %v236, i64 %v227
  %v238 = load float, ptr %v237, align 4
  %v239 = add i64 %v270, %v225
  %v240 = extractvalue { ptr, i64 } %v39, 1
  %v241 = icmp ult i64 %v239, %v240
  br i1 %v241, label %bb46, label %bb84
bb46:
  %v242 = extractvalue { ptr, i64 } %v39, 0
  %v243 = getelementptr inbounds i16, ptr %v242, i64 %v239
  %v244 = load i16, ptr %v243, align 2
  %v245 = zext i16 %v244 to i32
  %v246 = and i32 16, 31
  %v247 = shl i32 %v245, %v246
  %v248 = bitcast i32 %v247 to float
  %v249 = fmul contract float %v248, %v269
  %v250 = fmul contract float %v249, %v253
  %v251 = fmul contract float %v238, %v250
  %v252 = fadd contract float %v221, %v251
  br label %bb42
bb47:
  ret void
bb48:
  %v253 = fdiv contract float 1.0, %v66
  %v254 = extractvalue { ptr, i64 } %v44, 1
  %v255 = icmp ult i64 %v62, %v254
  br i1 %v255, label %bb6, label %bb85
bb49:
  %v256 = add i64 %v113, 1
  %v257 = insertvalue { i64, i64 } undef, i64 1, 0
  %v258 = insertvalue { i64, i64 } %v257, i64 %v113, 1
  br label %bb51
bb50:
  %v259 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb51
bb51:
  %v260 = phi { i64, i64 } [ %v258, %bb49 ], [ %v259, %bb50 ]
  %v261 = phi i64 [ %v256, %bb49 ], [ %v113, %bb50 ]
  %v262 = extractvalue { i64, i64 } %v260, 0
  %v263 = bitcast i64 %v262 to i64
  %v264 = icmp eq i64 %v263, 0
  br i1 %v264, label %bb24, label %bb52
bb52:
  %v265 = icmp eq i64 %v263, 1
  br i1 %v265, label %bb23, label %bb22
bb53:
  %v266 = fdiv contract float 1.0, %v122
  %v267 = fadd contract float %v112, 0.0000009999999974752427
  %v268 = call float @__nv_sqrtf(float %v267) #0
  br label %bb54
bb54:
  %v269 = fdiv contract float 1.0, %v268
  %v270 = mul i64 %v62, %v64
  %v271 = mul i64 %v270, %v59
  %v272 = add i64 %v271, %v63
  br label %bb27
bb55:
  %v273 = add i64 %v143, 1
  %v274 = insertvalue { i64, i64 } undef, i64 1, 0
  %v275 = insertvalue { i64, i64 } %v274, i64 %v143, 1
  br label %bb57
bb56:
  %v276 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb57
bb57:
  %v277 = phi { i64, i64 } [ %v275, %bb55 ], [ %v276, %bb56 ]
  %v278 = phi i64 [ %v273, %bb55 ], [ %v143, %bb56 ]
  %v279 = extractvalue { i64, i64 } %v277, 0
  %v280 = bitcast i64 %v279 to i64
  %v281 = icmp eq i64 %v280, 0
  br i1 %v281, label %bb29, label %bb58
bb58:
  %v282 = icmp eq i64 %v280, 1
  br i1 %v282, label %bb28, label %bb22
bb59:
  %v283 = add i64 %v158, 1
  %v284 = insertvalue { i64, i64 } undef, i64 1, 0
  %v285 = insertvalue { i64, i64 } %v284, i64 %v158, 1
  br label %bb61
bb60:
  %v286 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb61
bb61:
  %v287 = phi { i64, i64 } [ %v285, %bb59 ], [ %v286, %bb60 ]
  %v288 = phi i64 [ %v283, %bb59 ], [ %v158, %bb60 ]
  %v289 = extractvalue { i64, i64 } %v287, 0
  %v290 = bitcast i64 %v289 to i64
  %v291 = icmp eq i64 %v290, 0
  br i1 %v291, label %bb33, label %bb62
bb62:
  %v292 = icmp eq i64 %v290, 1
  br i1 %v292, label %bb32, label %bb22
bb63:
  %v293 = add i64 %v195, 1
  %v294 = insertvalue { i64, i64 } undef, i64 1, 0
  %v295 = insertvalue { i64, i64 } %v294, i64 %v195, 1
  br label %bb65
bb64:
  %v296 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb65
bb65:
  %v297 = phi { i64, i64 } [ %v295, %bb63 ], [ %v296, %bb64 ]
  %v298 = phi i64 [ %v293, %bb63 ], [ %v195, %bb64 ]
  %v299 = extractvalue { i64, i64 } %v297, 0
  %v300 = bitcast i64 %v299 to i64
  %v301 = icmp eq i64 %v300, 0
  br i1 %v301, label %bb39, label %bb66
bb66:
  %v302 = icmp eq i64 %v300, 1
  br i1 %v302, label %bb38, label %bb22
bb67:
  %v303 = add i64 %v222, 1
  %v304 = insertvalue { i64, i64 } undef, i64 1, 0
  %v305 = insertvalue { i64, i64 } %v304, i64 %v222, 1
  br label %bb69
bb68:
  %v306 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb69
bb69:
  %v307 = phi { i64, i64 } [ %v305, %bb67 ], [ %v306, %bb68 ]
  %v308 = phi i64 [ %v303, %bb67 ], [ %v222, %bb68 ]
  %v309 = extractvalue { i64, i64 } %v307, 0
  %v310 = bitcast i64 %v309 to i64
  %v311 = icmp eq i64 %v310, 0
  br i1 %v311, label %bb44, label %bb70
bb70:
  %v312 = icmp eq i64 %v310, 1
  br i1 %v312, label %bb43, label %bb22
bb71:
  unreachable
bb72:
  unreachable
bb73:
  unreachable
bb74:
  unreachable
bb75:
  unreachable
bb76:
  unreachable
bb77:
  unreachable
bb78:
  unreachable
bb79:
  unreachable
bb80:
  unreachable
bb81:
  unreachable
bb82:
  unreachable
bb83:
  unreachable
bb84:
  unreachable
bb85:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()

define void @infers_rope_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v7, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v9, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v15, %entry ]
  %v25 = phi { ptr, i64 } [ %v17, %entry ]
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi { ptr, i64 } [ %v23, %entry ]
  %v29 = phi i32 [ %v10, %entry ]
  %v30 = phi i32 [ %v11, %entry ]
  %v31 = phi i32 [ %v12, %entry ]
  %v32 = phi i32 [ %v13, %entry ]
  %v33 = alloca {  }, align 1
  %v34 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v36 = udiv i32 %v32, 2
  %v37 = zext i32 %v36 to i64
  %v38 = zext i32 %v29 to i64
  %v39 = zext i32 %v30 to i64
  %v40 = mul i64 %v38, %v39
  %v41 = mul i64 %v40, %v37
  %v42 = bitcast ptr %v33 to ptr
  %v43 = call i64 @cuda_device____internal__index_1d(ptr %v42) #0
  br label %bb2
bb2:
  %v44 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v45 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v46 = mul i32 %v44, %v45
  %v47 = insertvalue { i64, i64 } undef, i64 %v43, 0
  %v48 = insertvalue { i64, i64 } %v47, i64 %v41, 1
  %v49 = zext i32 %v46 to i64
  %v50 = extractvalue { i64, i64 } %v48, 0
  %v51 = extractvalue { i64, i64 } %v48, 1
  %v52 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v50, i64 %v51, i64 %v49) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v52, ptr %v34, align 8
  br label %bb15
bb5:
  %v53 = phi i64 [ %v177, %bb14 ], [ %v163, %bb15 ]
  %v54 = phi i64 [ %v178, %bb14 ], [ %v166, %bb15 ]
  %v55 = add i64 %v168, 1
  %v56 = icmp eq i64 %v55, 0
  %v57 = select i1 %v56, i8 0, i8 1
  %v58 = insertvalue { i8, { { i64 } } } undef, i8 %v57, 0
  %v59 = insertvalue { i8, { { i64 } } } %v58, i64 %v55, 1, 0, 0
  %v60 = extractvalue { i8, { { i64 } } } %v59, 0
  %v61 = zext i8 %v60 to i64
  %v62 = icmp eq i64 %v61, 1
  %v63 = extractvalue { i8, { { i64 } } } %v59, 1
  %v64 = alloca { { i64 } }, align 8
  store { { i64 } } %v63, ptr %v64, align 8
  %v65 = load i64, ptr %v64, align 8
  %v66 = icmp ugt i64 %v54, 0
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb17, label %bb16
bb6:
  unreachable
bb7:
  %v68 = extractvalue { i64, i64 } %v176, 1
  %v69 = mul i32 %v30, %v32
  %v70 = udiv i32 %v69, 2
  %v71 = zext i32 %v70 to i64
  %v72 = icmp eq i64 %v71, 0
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb9, label %bb20
bb8:
  ret void
bb9:
  %v74 = udiv i64 %v68, %v71
  %v75 = icmp eq i64 %v37, 0
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb10, label %bb21
bb10:
  %v77 = udiv i64 %v68, %v37
  %v78 = icmp eq i64 %v39, 0
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb11, label %bb22
bb11:
  %v80 = urem i64 %v77, %v39
  %v81 = urem i64 %v68, %v37
  %v82 = extractvalue { ptr, i64 } %v28, 1
  %v83 = icmp ult i64 %v74, %v82
  br i1 %v83, label %bb12, label %bb23
bb12:
  %v84 = extractvalue { ptr, i64 } %v28, 0
  %v85 = getelementptr inbounds i32, ptr %v84, i64 %v74
  %v86 = load i32, ptr %v85, align 4
  %v87 = sext i32 %v86 to i64
  %v88 = mul i64 %v87, %v37
  %v89 = add i64 %v88, %v81
  %v90 = extractvalue { ptr, i64 } %v26, 1
  %v91 = icmp ult i64 %v89, %v90
  br i1 %v91, label %bb13, label %bb24
bb13:
  %v92 = extractvalue { ptr, i64 } %v26, 0
  %v93 = getelementptr inbounds float, ptr %v92, i64 %v89
  %v94 = load float, ptr %v93, align 4
  %v95 = extractvalue { ptr, i64 } %v27, 1
  %v96 = icmp ult i64 %v89, %v95
  br i1 %v96, label %bb14, label %bb25
bb14:
  %v97 = extractvalue { ptr, i64 } %v27, 0
  %v98 = getelementptr inbounds float, ptr %v97, i64 %v89
  %v99 = load float, ptr %v98, align 4
  %v100 = mul i64 %v74, %v39
  %v101 = zext i32 %v31 to i64
  %v102 = mul i64 %v100, %v101
  %v103 = mul i64 %v80, %v101
  %v104 = add i64 %v102, %v103
  %v105 = add i64 %v104, %v81
  %v106 = add i64 %v105, %v37
  %v107 = extractvalue { ptr, i64 } %v24, 0
  %v108 = getelementptr inbounds i16, ptr %v107, i64 %v105
  %v109 = load i16, ptr %v108, align 2
  %v110 = zext i16 %v109 to i32
  %v111 = and i32 16, 31
  %v112 = shl i32 %v110, %v111
  %v113 = bitcast i32 %v112 to float
  %v114 = getelementptr inbounds i16, ptr %v107, i64 %v106
  %v115 = load i16, ptr %v114, align 2
  %v116 = zext i16 %v115 to i32
  %v117 = and i32 16, 31
  %v118 = shl i32 %v116, %v117
  %v119 = bitcast i32 %v118 to float
  %v120 = fmul contract float %v113, %v94
  %v121 = fmul contract float %v119, %v99
  %v122 = fsub contract float %v120, %v121
  %v123 = bitcast float %v122 to i32
  %v124 = and i32 16, 31
  %v125 = lshr i32 %v123, %v124
  %v126 = trunc i32 %v125 to i16
  store i16 %v126, ptr %v108, align 2
  %v127 = fmul contract float %v113, %v99
  %v128 = fmul contract float %v119, %v94
  %v129 = fadd contract float %v127, %v128
  %v130 = bitcast float %v129 to i32
  %v131 = and i32 16, 31
  %v132 = lshr i32 %v130, %v131
  %v133 = trunc i32 %v132 to i16
  store i16 %v133, ptr %v114, align 2
  %v134 = extractvalue { ptr, i64 } %v25, 0
  %v135 = getelementptr inbounds i16, ptr %v134, i64 %v105
  %v136 = load i16, ptr %v135, align 2
  %v137 = zext i16 %v136 to i32
  %v138 = and i32 16, 31
  %v139 = shl i32 %v137, %v138
  %v140 = bitcast i32 %v139 to float
  %v141 = getelementptr inbounds i16, ptr %v134, i64 %v106
  %v142 = load i16, ptr %v141, align 2
  %v143 = zext i16 %v142 to i32
  %v144 = and i32 16, 31
  %v145 = shl i32 %v143, %v144
  %v146 = bitcast i32 %v145 to float
  %v147 = fmul contract float %v140, %v94
  %v148 = fmul contract float %v146, %v99
  %v149 = fsub contract float %v147, %v148
  %v150 = bitcast float %v149 to i32
  %v151 = and i32 16, 31
  %v152 = lshr i32 %v150, %v151
  %v153 = trunc i32 %v152 to i16
  store i16 %v153, ptr %v135, align 2
  %v154 = fmul contract float %v140, %v99
  %v155 = fmul contract float %v146, %v94
  %v156 = fadd contract float %v154, %v155
  %v157 = bitcast float %v156 to i32
  %v158 = and i32 16, 31
  %v159 = lshr i32 %v157, %v158
  %v160 = trunc i32 %v159 to i16
  store i16 %v160, ptr %v141, align 2
  br label %bb5
bb15:
  %v161 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v162 = getelementptr inbounds { i64, i64 }, ptr %v161, i32 0, i32 0
  %v163 = load i64, ptr %v162, align 8
  %v164 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v165 = getelementptr inbounds { i64, i64 }, ptr %v164, i32 0, i32 1
  %v166 = load i64, ptr %v165, align 8
  %v167 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 1
  %v168 = load i64, ptr %v167, align 8
  %v169 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 2
  %v170 = load i1, ptr %v169, align 1
  br label %bb5
bb16:
  %v171 = add i64 %v53, %v65
  %v172 = sub i64 %v54, 1
  %v173 = insertvalue { i64, i64 } undef, i64 1, 0
  %v174 = insertvalue { i64, i64 } %v173, i64 %v53, 1
  br label %bb18
bb17:
  %v175 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb18
bb18:
  %v176 = phi { i64, i64 } [ %v174, %bb16 ], [ %v175, %bb17 ]
  %v177 = phi i64 [ %v171, %bb16 ], [ %v53, %bb17 ]
  %v178 = phi i64 [ %v172, %bb16 ], [ %v54, %bb17 ]
  %v179 = extractvalue { i64, i64 } %v176, 0
  %v180 = bitcast i64 %v179 to i64
  %v181 = icmp eq i64 %v180, 0
  br i1 %v181, label %bb8, label %bb19
bb19:
  %v182 = icmp eq i64 %v180, 1
  br i1 %v182, label %bb7, label %bb6
bb20:
  unreachable
bb21:
  unreachable
bb22:
  unreachable
bb23:
  unreachable
bb24:
  unreachable
bb25:
  unreachable
}

define void @infers_paged_kv_write_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, ptr %v8, i64 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13) #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v1, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v3, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v5, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v7, 1
  %v22 = insertvalue { ptr, i64 } undef, ptr %v8, 0
  %v23 = insertvalue { ptr, i64 } %v22, i64 %v9, 1
  br label %bb0
bb0:
  %v24 = phi { ptr, i64 } [ %v15, %entry ]
  %v25 = phi { ptr, i64 } [ %v17, %entry ]
  %v26 = phi { ptr, i64 } [ %v19, %entry ]
  %v27 = phi { ptr, i64 } [ %v21, %entry ]
  %v28 = phi { ptr, i64 } [ %v23, %entry ]
  %v29 = phi i32 [ %v10, %entry ]
  %v30 = phi i32 [ %v11, %entry ]
  %v31 = phi i32 [ %v12, %entry ]
  %v32 = phi i32 [ %v13, %entry ]
  %v33 = alloca {  }, align 1
  %v34 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v36 = bitcast ptr %v33 to ptr
  %v37 = call i64 @cuda_device____internal__index_1d(ptr %v36) #0
  br label %bb2
bb2:
  %v38 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v40 = mul i32 %v38, %v39
  %v41 = zext i32 %v29 to i64
  %v42 = zext i32 %v32 to i64
  %v43 = mul i64 %v41, %v42
  %v44 = insertvalue { i64, i64 } undef, i64 %v37, 0
  %v45 = insertvalue { i64, i64 } %v44, i64 %v43, 1
  %v46 = zext i32 %v40 to i64
  %v47 = extractvalue { i64, i64 } %v45, 0
  %v48 = extractvalue { i64, i64 } %v45, 1
  %v49 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v47, i64 %v48, i64 %v46) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v49, ptr %v34, align 8
  br label %bb15
bb5:
  %v50 = phi i64 [ %v126, %bb14 ], [ %v112, %bb15 ]
  %v51 = phi i64 [ %v127, %bb14 ], [ %v115, %bb15 ]
  %v52 = add i64 %v117, 1
  %v53 = icmp eq i64 %v52, 0
  %v54 = select i1 %v53, i8 0, i8 1
  %v55 = insertvalue { i8, { { i64 } } } undef, i8 %v54, 0
  %v56 = insertvalue { i8, { { i64 } } } %v55, i64 %v52, 1, 0, 0
  %v57 = extractvalue { i8, { { i64 } } } %v56, 0
  %v58 = zext i8 %v57 to i64
  %v59 = icmp eq i64 %v58, 1
  %v60 = extractvalue { i8, { { i64 } } } %v56, 1
  %v61 = alloca { { i64 } }, align 8
  store { { i64 } } %v60, ptr %v61, align 8
  %v62 = load i64, ptr %v61, align 8
  %v63 = icmp ugt i64 %v51, 0
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb17, label %bb16
bb6:
  unreachable
bb7:
  %v65 = extractvalue { i64, i64 } %v125, 1
  %v66 = icmp eq i64 %v42, 0
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb9, label %bb20
bb8:
  ret void
bb9:
  %v68 = udiv i64 %v65, %v42
  %v69 = urem i64 %v65, %v42
  %v70 = extractvalue { ptr, i64 } %v28, 1
  %v71 = icmp ult i64 %v68, %v70
  br i1 %v71, label %bb10, label %bb21
bb10:
  %v72 = extractvalue { ptr, i64 } %v28, 0
  %v73 = getelementptr inbounds i32, ptr %v72, i64 %v68
  %v74 = load i32, ptr %v73, align 4
  %v75 = sext i32 %v74 to i64
  %v76 = zext i32 %v31 to i64
  %v77 = icmp eq i64 %v76, 0
  %v78 = xor i1 %v77, 1
  br i1 %v78, label %bb11, label %bb22
bb11:
  %v79 = udiv i64 %v75, %v76
  %v80 = urem i64 %v75, %v76
  %v81 = extractvalue { ptr, i64 } %v27, 1
  %v82 = icmp ult i64 %v79, %v81
  br i1 %v82, label %bb12, label %bb23
bb12:
  %v83 = extractvalue { ptr, i64 } %v27, 0
  %v84 = getelementptr inbounds i32, ptr %v83, i64 %v79
  %v85 = load i32, ptr %v84, align 4
  %v86 = sext i32 %v85 to i64
  %v87 = mul i64 2, %v76
  %v88 = mul i64 %v87, %v42
  %v89 = mul i64 %v86, %v88
  %v90 = mul i64 %v80, %v42
  %v91 = add i64 %v89, %v90
  %v92 = add i64 %v91, %v69
  %v93 = extractvalue { ptr, i64 } %v24, 1
  %v94 = icmp ult i64 %v65, %v93
  br i1 %v94, label %bb13, label %bb24
bb13:
  %v95 = extractvalue { ptr, i64 } %v24, 0
  %v96 = getelementptr inbounds i16, ptr %v95, i64 %v65
  %v97 = load i16, ptr %v96, align 2
  %v98 = extractvalue { ptr, i64 } %v26, 0
  %v99 = getelementptr inbounds i16, ptr %v98, i64 %v92
  store i16 %v97, ptr %v99, align 2
  %v100 = mul i64 %v76, %v42
  %v101 = add i64 %v89, %v100
  %v102 = add i64 %v101, %v90
  %v103 = add i64 %v102, %v69
  %v104 = extractvalue { ptr, i64 } %v25, 1
  %v105 = icmp ult i64 %v65, %v104
  br i1 %v105, label %bb14, label %bb25
bb14:
  %v106 = extractvalue { ptr, i64 } %v25, 0
  %v107 = getelementptr inbounds i16, ptr %v106, i64 %v65
  %v108 = load i16, ptr %v107, align 2
  %v109 = getelementptr inbounds i16, ptr %v98, i64 %v103
  store i16 %v108, ptr %v109, align 2
  br label %bb5
bb15:
  %v110 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v111 = getelementptr inbounds { i64, i64 }, ptr %v110, i32 0, i32 0
  %v112 = load i64, ptr %v111, align 8
  %v113 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v114 = getelementptr inbounds { i64, i64 }, ptr %v113, i32 0, i32 1
  %v115 = load i64, ptr %v114, align 8
  %v116 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 1
  %v117 = load i64, ptr %v116, align 8
  %v118 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 2
  %v119 = load i1, ptr %v118, align 1
  br label %bb5
bb16:
  %v120 = add i64 %v50, %v62
  %v121 = sub i64 %v51, 1
  %v122 = insertvalue { i64, i64 } undef, i64 1, 0
  %v123 = insertvalue { i64, i64 } %v122, i64 %v50, 1
  br label %bb18
bb17:
  %v124 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb18
bb18:
  %v125 = phi { i64, i64 } [ %v123, %bb16 ], [ %v124, %bb17 ]
  %v126 = phi i64 [ %v120, %bb16 ], [ %v50, %bb17 ]
  %v127 = phi i64 [ %v121, %bb16 ], [ %v51, %bb17 ]
  %v128 = extractvalue { i64, i64 } %v125, 0
  %v129 = bitcast i64 %v128 to i64
  %v130 = icmp eq i64 %v129, 0
  br i1 %v130, label %bb8, label %bb19
bb19:
  %v131 = icmp eq i64 %v129, 1
  br i1 %v131, label %bb7, label %bb6
bb20:
  unreachable
bb21:
  unreachable
bb22:
  unreachable
bb23:
  unreachable
bb24:
  unreachable
bb25:
  unreachable
}

declare float @__nv_fmaxf(float, float)

define void @infers_paged_attention_decode_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, ptr %v13, i64 %v14) #0 {
entry:
  %v15 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v1, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v3, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v5, 1
  %v21 = insertvalue { ptr, i64 } undef, ptr %v13, 0
  %v22 = insertvalue { ptr, i64 } %v21, i64 %v14, 1
  br label %bb0
bb0:
  %v23 = phi { ptr, i64 } [ %v16, %entry ]
  %v24 = phi { ptr, i64 } [ %v18, %entry ]
  %v25 = phi { ptr, i64 } [ %v20, %entry ]
  %v26 = phi i32 [ %v6, %entry ]
  %v27 = phi i32 [ %v7, %entry ]
  %v28 = phi i32 [ %v8, %entry ]
  %v29 = phi i32 [ %v9, %entry ]
  %v30 = phi i32 [ %v10, %entry ]
  %v31 = phi i32 [ %v11, %entry ]
  %v32 = phi i32 [ %v12, %entry ]
  %v33 = phi { ptr, i64 } [ %v22, %entry ]
  %v34 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v37 = zext i32 %v36 to i64
  %v38 = zext i32 %v29 to i64
  %v39 = icmp uge i64 %v37, %v38
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb4, label %bb3
bb3:
  br label %bb59
bb4:
  %v41 = icmp eq i32 %v29, 0
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb5, label %bb87
bb5:
  %v43 = udiv i32 %v30, %v29
  %v44 = zext i32 %v43 to i64
  %v45 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb6
bb6:
  %v46 = zext i32 %v45 to i64
  %v47 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb7
bb7:
  %v48 = zext i32 %v47 to i64
  %v49 = zext i32 %v31 to i64
  %v50 = mul i64 2, %v49
  %v51 = zext i32 %v32 to i64
  %v52 = mul i64 %v50, %v51
  %v53 = uitofp i32 %v28 to float
  %v54 = call float @dev_sqrtf(float %v53) #0
  br label %bb60
bb8:
  br label %bb9
bb9:
  %v55 = phi i64 [ 0, %bb8 ], [ %v249, %bb58 ]
  %v56 = icmp ult i64 %v55, %v44
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb62, label %bb61
bb10:
  unreachable
bb11:
  %v58 = extractvalue { i64, i64 } %v248, 1
  %v59 = mul i64 %v37, %v44
  %v60 = add i64 %v59, %v58
  %v61 = zext i32 %v28 to i64
  %v62 = icmp ult i64 %v46, %v61
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb15, label %bb13
bb12:
  br label %bb59
bb13:
  %v64 = mul i64 %v60, %v61
  %v65 = add i64 %v64, %v46
  %v66 = extractvalue { ptr, i64 } %v23, 1
  %v67 = icmp ult i64 %v65, %v66
  br i1 %v67, label %bb14, label %bb88
bb14:
  %v68 = extractvalue { ptr, i64 } %v23, 0
  %v69 = getelementptr inbounds i16, ptr %v68, i64 %v65
  %v70 = load i16, ptr %v69, align 2
  %v71 = zext i16 %v70 to i32
  %v72 = and i32 16, 31
  %v73 = shl i32 %v71, %v72
  %v74 = bitcast i32 %v73 to float
  %v75 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v46
  %v76 = addrspacecast ptr addrspace(3) %v75 to ptr
  store float %v74, ptr %v76, align 4
  br label %bb15
bb15:
  call void @llvm.nvvm.barrier0() #0
  br label %bb16
bb16:
  %v78 = zext i32 %v27 to i64
  %v79 = insertvalue { i64, i64 } undef, i64 %v46, 0
  %v80 = insertvalue { i64, i64 } %v79, i64 %v78, 1
  %v81 = extractvalue { i64, i64 } %v80, 0
  %v82 = extractvalue { i64, i64 } %v80, 1
  %v83 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v81, i64 %v82, i64 %v48) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v83, ptr %v34, align 8
  br label %bb65
bb17:
  %v84 = phi float [ %v133, %bb26 ], [ 0xFFF0000000000000, %bb65 ]
  %v85 = phi float [ %v137, %bb26 ], [ 0.0, %bb65 ]
  %v86 = phi i64 [ %v270, %bb26 ], [ %v256, %bb65 ]
  %v87 = phi i64 [ %v271, %bb26 ], [ %v259, %bb65 ]
  %v88 = add i64 %v261, 1
  %v89 = icmp eq i64 %v88, 0
  %v90 = select i1 %v89, i8 0, i8 1
  %v91 = insertvalue { i8, { { i64 } } } undef, i8 %v90, 0
  %v92 = insertvalue { i8, { { i64 } } } %v91, i64 %v88, 1, 0, 0
  %v93 = extractvalue { i8, { { i64 } } } %v92, 0
  %v94 = zext i8 %v93 to i64
  %v95 = icmp eq i64 %v94, 1
  %v96 = extractvalue { i8, { { i64 } } } %v92, 1
  %v97 = alloca { { i64 } }, align 8
  store { { i64 } } %v96, ptr %v97, align 8
  %v98 = load i64, ptr %v97, align 8
  %v99 = icmp ugt i64 %v87, 0
  %v100 = xor i1 %v99, 1
  br i1 %v100, label %bb67, label %bb66
bb18:
  %v101 = extractvalue { i64, i64 } %v269, 1
  %v102 = icmp eq i64 %v49, 0
  %v103 = xor i1 %v102, 1
  br i1 %v103, label %bb20, label %bb89
bb19:
  %v104 = add i64 %v48, %v46
  %v105 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v104
  %v106 = addrspacecast ptr addrspace(3) %v105 to ptr
  store float %v84, ptr %v106, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb27
bb20:
  %v108 = udiv i64 %v101, %v49
  %v109 = urem i64 %v101, %v49
  %v110 = extractvalue { ptr, i64 } %v25, 1
  %v111 = icmp ult i64 %v108, %v110
  br i1 %v111, label %bb21, label %bb90
bb21:
  %v112 = extractvalue { ptr, i64 } %v25, 0
  %v113 = getelementptr inbounds i32, ptr %v112, i64 %v108
  %v114 = load i32, ptr %v113, align 4
  %v115 = sext i32 %v114 to i64
  br label %bb22
bb22:
  %v116 = phi float [ 0.0, %bb21 ], [ %v296, %bb75 ]
  %v117 = phi i64 [ 0, %bb21 ], [ %v281, %bb75 ]
  %v118 = icmp ult i64 %v117, %v61
  %v119 = xor i1 %v118, 1
  br i1 %v119, label %bb71, label %bb70
bb23:
  %v120 = extractvalue { i64, i64 } %v280, 1
  %v121 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v120
  %v122 = addrspacecast ptr addrspace(3) %v121 to ptr
  %v123 = load float, ptr %v122, align 4
  %v124 = mul i64 %v115, %v52
  %v125 = mul i64 %v109, %v51
  %v126 = add i64 %v124, %v125
  %v127 = mul i64 %v37, %v61
  %v128 = add i64 %v126, %v127
  %v129 = add i64 %v128, %v120
  %v130 = extractvalue { ptr, i64 } %v24, 1
  %v131 = icmp ult i64 %v129, %v130
  br i1 %v131, label %bb75, label %bb91
bb24:
  %v132 = fmul contract float %v116, %v243
  %v133 = call float @__nv_fmaxf(float %v84, float %v132) #0
  br label %bb74
bb25:
  %v134 = fmul contract float %v85, %v287
  %v135 = fsub contract float %v132, %v133
  %v136 = call float @__nv_expf(float %v135) #0
  br label %bb26
bb26:
  %v137 = fadd contract float %v134, %v136
  br label %bb17
bb27:
  %v138 = udiv i64 %v48, 2
  br label %bb28
bb28:
  %v139 = phi i64 [ %v138, %bb27 ], [ %v153, %bb33 ]
  %v140 = icmp ugt i64 %v139, 0
  %v141 = xor i1 %v140, 1
  br i1 %v141, label %bb34, label %bb29
bb29:
  %v142 = icmp ult i64 %v46, %v139
  %v143 = xor i1 %v142, 1
  br i1 %v143, label %bb31, label %bb30
bb30:
  %v144 = load float, ptr %v106, align 4
  %v145 = add i64 %v104, %v139
  %v146 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v145
  %v147 = addrspacecast ptr addrspace(3) %v146 to ptr
  %v148 = load float, ptr %v147, align 4
  %v149 = call float @__nv_fmaxf(float %v144, float %v148) #0
  br label %bb76
bb31:
  br label %bb32
bb32:
  call void @llvm.nvvm.barrier0() #0
  br label %bb33
bb33:
  %v151 = zext i32 1 to i64
  %v152 = and i64 %v151, 63
  %v153 = lshr i64 %v139, %v152
  br label %bb28
bb34:
  %v154 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v48
  %v155 = addrspacecast ptr addrspace(3) %v154 to ptr
  %v156 = load float, ptr %v155, align 4
  %v157 = fsub contract float %v84, %v156
  %v158 = call float @__nv_expf(float %v157) #0
  br label %bb35
bb35:
  %v159 = fmul contract float %v85, %v158
  %v160 = mul i64 2, %v48
  %v161 = add i64 %v160, %v46
  %v162 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v161
  %v163 = addrspacecast ptr addrspace(3) %v162 to ptr
  store float %v159, ptr %v163, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb36
bb36:
  %v165 = udiv i64 %v48, 2
  br label %bb37
bb37:
  %v166 = phi i64 [ %v165, %bb36 ], [ %v180, %bb42 ]
  %v167 = icmp ugt i64 %v166, 0
  %v168 = xor i1 %v167, 1
  br i1 %v168, label %bb43, label %bb38
bb38:
  %v169 = icmp ult i64 %v46, %v166
  %v170 = xor i1 %v169, 1
  br i1 %v170, label %bb40, label %bb39
bb39:
  %v171 = load float, ptr %v163, align 4
  %v172 = add i64 %v161, %v166
  %v173 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v172
  %v174 = addrspacecast ptr addrspace(3) %v173 to ptr
  %v175 = load float, ptr %v174, align 4
  %v176 = fadd contract float %v171, %v175
  store float %v176, ptr %v163, align 4
  br label %bb41
bb40:
  br label %bb41
bb41:
  call void @llvm.nvvm.barrier0() #0
  br label %bb42
bb42:
  %v178 = zext i32 1 to i64
  %v179 = and i64 %v178, 63
  %v180 = lshr i64 %v166, %v179
  br label %bb37
bb43:
  %v181 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v160
  %v182 = addrspacecast ptr addrspace(3) %v181 to ptr
  %v183 = load float, ptr %v182, align 4
  %v184 = fcmp ogt float %v183, 0.0
  %v185 = xor i1 %v184, 1
  br i1 %v185, label %bb45, label %bb44
bb44:
  %v186 = fdiv contract float 1.0, %v183
  br label %bb46
bb45:
  br label %bb46
bb46:
  %v187 = phi float [ %v186, %bb44 ], [ 0.0, %bb45 ]
  %v188 = xor i1 %v62, 1
  br i1 %v188, label %bb57, label %bb47
bb47:
  br label %bb48
bb48:
  %v189 = phi float [ 0.0, %bb47 ], [ %v334, %bb86 ]
  %v190 = phi i64 [ 0, %bb47 ], [ %v302, %bb86 ]
  %v191 = icmp ult i64 %v190, %v78
  %v192 = xor i1 %v191, 1
  br i1 %v192, label %bb78, label %bb77
bb49:
  %v193 = extractvalue { i64, i64 } %v301, 1
  %v194 = icmp eq i64 %v49, 0
  %v195 = xor i1 %v194, 1
  br i1 %v195, label %bb51, label %bb92
bb50:
  %v196 = bitcast float %v189 to i32
  %v197 = and i32 16, 31
  %v198 = lshr i32 %v196, %v197
  %v199 = trunc i32 %v198 to i16
  %v200 = mul i64 %v60, %v61
  %v201 = add i64 %v200, %v46
  %v202 = extractvalue { ptr, i64 } %v33, 0
  %v203 = getelementptr inbounds i16, ptr %v202, i64 %v201
  store i16 %v199, ptr %v203, align 2
  br label %bb57
bb51:
  %v204 = udiv i64 %v193, %v49
  %v205 = urem i64 %v193, %v49
  %v206 = extractvalue { ptr, i64 } %v25, 1
  %v207 = icmp ult i64 %v204, %v206
  br i1 %v207, label %bb52, label %bb93
bb52:
  %v208 = extractvalue { ptr, i64 } %v25, 0
  %v209 = getelementptr inbounds i32, ptr %v208, i64 %v204
  %v210 = load i32, ptr %v209, align 4
  %v211 = sext i32 %v210 to i64
  br label %bb53
bb53:
  %v212 = phi float [ 0.0, %bb52 ], [ %v325, %bb85 ]
  %v213 = phi i64 [ 0, %bb52 ], [ %v312, %bb85 ]
  %v214 = icmp ult i64 %v213, %v61
  %v215 = xor i1 %v214, 1
  br i1 %v215, label %bb82, label %bb81
bb54:
  %v216 = extractvalue { i64, i64 } %v311, 1
  %v217 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_paged_attention_decode_bf16, i64 %v216
  %v218 = addrspacecast ptr addrspace(3) %v217 to ptr
  %v219 = load float, ptr %v218, align 4
  %v220 = mul i64 %v211, %v52
  %v221 = mul i64 %v205, %v51
  %v222 = add i64 %v220, %v221
  %v223 = mul i64 %v37, %v61
  %v224 = add i64 %v222, %v223
  %v225 = add i64 %v224, %v216
  %v226 = extractvalue { ptr, i64 } %v24, 1
  %v227 = icmp ult i64 %v225, %v226
  br i1 %v227, label %bb85, label %bb94
bb55:
  %v228 = fmul contract float %v212, %v243
  %v229 = fsub contract float %v228, %v156
  %v230 = call float @__nv_expf(float %v229) #0
  br label %bb56
bb56:
  %v231 = fmul contract float %v230, %v187
  %v232 = mul i64 %v211, %v52
  %v233 = mul i64 %v49, %v51
  %v234 = add i64 %v232, %v233
  %v235 = mul i64 %v205, %v51
  %v236 = add i64 %v234, %v235
  %v237 = mul i64 %v37, %v61
  %v238 = add i64 %v236, %v237
  %v239 = add i64 %v238, %v46
  %v240 = extractvalue { ptr, i64 } %v24, 1
  %v241 = icmp ult i64 %v239, %v240
  br i1 %v241, label %bb86, label %bb95
bb57:
  call void @llvm.nvvm.barrier0() #0
  br label %bb58
bb58:
  br label %bb9
bb59:
  ret void
bb60:
  %v243 = fdiv contract float 1.0, %v54
  br label %bb8
bb61:
  %v244 = add i64 %v55, 1
  %v245 = insertvalue { i64, i64 } undef, i64 1, 0
  %v246 = insertvalue { i64, i64 } %v245, i64 %v55, 1
  br label %bb63
bb62:
  %v247 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb63
bb63:
  %v248 = phi { i64, i64 } [ %v246, %bb61 ], [ %v247, %bb62 ]
  %v249 = phi i64 [ %v244, %bb61 ], [ %v55, %bb62 ]
  %v250 = extractvalue { i64, i64 } %v248, 0
  %v251 = bitcast i64 %v250 to i64
  %v252 = icmp eq i64 %v251, 0
  br i1 %v252, label %bb12, label %bb64
bb64:
  %v253 = icmp eq i64 %v251, 1
  br i1 %v253, label %bb11, label %bb10
bb65:
  %v254 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v255 = getelementptr inbounds { i64, i64 }, ptr %v254, i32 0, i32 0
  %v256 = load i64, ptr %v255, align 8
  %v257 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 0
  %v258 = getelementptr inbounds { i64, i64 }, ptr %v257, i32 0, i32 1
  %v259 = load i64, ptr %v258, align 8
  %v260 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 1
  %v261 = load i64, ptr %v260, align 8
  %v262 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v34, i32 0, i32 2
  %v263 = load i1, ptr %v262, align 1
  br label %bb17
bb66:
  %v264 = add i64 %v86, %v98
  %v265 = sub i64 %v87, 1
  %v266 = insertvalue { i64, i64 } undef, i64 1, 0
  %v267 = insertvalue { i64, i64 } %v266, i64 %v86, 1
  br label %bb68
bb67:
  %v268 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb68
bb68:
  %v269 = phi { i64, i64 } [ %v267, %bb66 ], [ %v268, %bb67 ]
  %v270 = phi i64 [ %v264, %bb66 ], [ %v86, %bb67 ]
  %v271 = phi i64 [ %v265, %bb66 ], [ %v87, %bb67 ]
  %v272 = extractvalue { i64, i64 } %v269, 0
  %v273 = bitcast i64 %v272 to i64
  %v274 = icmp eq i64 %v273, 0
  br i1 %v274, label %bb19, label %bb69
bb69:
  %v275 = icmp eq i64 %v273, 1
  br i1 %v275, label %bb18, label %bb10
bb70:
  %v276 = add i64 %v117, 1
  %v277 = insertvalue { i64, i64 } undef, i64 1, 0
  %v278 = insertvalue { i64, i64 } %v277, i64 %v117, 1
  br label %bb72
bb71:
  %v279 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb72
bb72:
  %v280 = phi { i64, i64 } [ %v278, %bb70 ], [ %v279, %bb71 ]
  %v281 = phi i64 [ %v276, %bb70 ], [ %v117, %bb71 ]
  %v282 = extractvalue { i64, i64 } %v280, 0
  %v283 = bitcast i64 %v282 to i64
  %v284 = icmp eq i64 %v283, 0
  br i1 %v284, label %bb24, label %bb73
bb73:
  %v285 = icmp eq i64 %v283, 1
  br i1 %v285, label %bb23, label %bb10
bb74:
  %v286 = fsub contract float %v84, %v133
  %v287 = call float @__nv_expf(float %v286) #0
  br label %bb25
bb75:
  %v288 = extractvalue { ptr, i64 } %v24, 0
  %v289 = getelementptr inbounds i16, ptr %v288, i64 %v129
  %v290 = load i16, ptr %v289, align 2
  %v291 = zext i16 %v290 to i32
  %v292 = and i32 16, 31
  %v293 = shl i32 %v291, %v292
  %v294 = bitcast i32 %v293 to float
  %v295 = fmul contract float %v123, %v294
  %v296 = fadd contract float %v116, %v295
  br label %bb22
bb76:
  store float %v149, ptr %v106, align 4
  br label %bb32
bb77:
  %v297 = add i64 %v190, 1
  %v298 = insertvalue { i64, i64 } undef, i64 1, 0
  %v299 = insertvalue { i64, i64 } %v298, i64 %v190, 1
  br label %bb79
bb78:
  %v300 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb79
bb79:
  %v301 = phi { i64, i64 } [ %v299, %bb77 ], [ %v300, %bb78 ]
  %v302 = phi i64 [ %v297, %bb77 ], [ %v190, %bb78 ]
  %v303 = extractvalue { i64, i64 } %v301, 0
  %v304 = bitcast i64 %v303 to i64
  %v305 = icmp eq i64 %v304, 0
  br i1 %v305, label %bb50, label %bb80
bb80:
  %v306 = icmp eq i64 %v304, 1
  br i1 %v306, label %bb49, label %bb10
bb81:
  %v307 = add i64 %v213, 1
  %v308 = insertvalue { i64, i64 } undef, i64 1, 0
  %v309 = insertvalue { i64, i64 } %v308, i64 %v213, 1
  br label %bb83
bb82:
  %v310 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb83
bb83:
  %v311 = phi { i64, i64 } [ %v309, %bb81 ], [ %v310, %bb82 ]
  %v312 = phi i64 [ %v307, %bb81 ], [ %v213, %bb82 ]
  %v313 = extractvalue { i64, i64 } %v311, 0
  %v314 = bitcast i64 %v313 to i64
  %v315 = icmp eq i64 %v314, 0
  br i1 %v315, label %bb55, label %bb84
bb84:
  %v316 = icmp eq i64 %v314, 1
  br i1 %v316, label %bb54, label %bb10
bb85:
  %v317 = extractvalue { ptr, i64 } %v24, 0
  %v318 = getelementptr inbounds i16, ptr %v317, i64 %v225
  %v319 = load i16, ptr %v318, align 2
  %v320 = zext i16 %v319 to i32
  %v321 = and i32 16, 31
  %v322 = shl i32 %v320, %v321
  %v323 = bitcast i32 %v322 to float
  %v324 = fmul contract float %v219, %v323
  %v325 = fadd contract float %v212, %v324
  br label %bb53
bb86:
  %v326 = extractvalue { ptr, i64 } %v24, 0
  %v327 = getelementptr inbounds i16, ptr %v326, i64 %v239
  %v328 = load i16, ptr %v327, align 2
  %v329 = zext i16 %v328 to i32
  %v330 = and i32 16, 31
  %v331 = shl i32 %v329, %v330
  %v332 = bitcast i32 %v331 to float
  %v333 = fmul contract float %v231, %v332
  %v334 = fadd contract float %v189, %v333
  br label %bb48
bb87:
  unreachable
bb88:
  unreachable
bb89:
  unreachable
bb90:
  unreachable
bb91:
  unreachable
bb92:
  unreachable
bb93:
  unreachable
bb94:
  unreachable
bb95:
  unreachable
}

define void @infers_paged_kv_read_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5, i32 %v6, i32 %v7, i32 %v8, ptr %v9, i64 %v10, ptr %v11, i64 %v12) #0 {
entry:
  %v13 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v9, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v10, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v11, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v12, 1
  br label %bb0
bb0:
  %v21 = phi { ptr, i64 } [ %v14, %entry ]
  %v22 = phi { ptr, i64 } [ %v16, %entry ]
  %v23 = phi i32 [ %v4, %entry ]
  %v24 = phi i32 [ %v5, %entry ]
  %v25 = phi i32 [ %v6, %entry ]
  %v26 = phi i32 [ %v7, %entry ]
  %v27 = phi i32 [ %v8, %entry ]
  %v28 = phi { ptr, i64 } [ %v18, %entry ]
  %v29 = phi { ptr, i64 } [ %v20, %entry ]
  %v30 = alloca {  }, align 1
  %v31 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v33 = bitcast ptr %v30 to ptr
  %v34 = call i64 @cuda_device____internal__index_1d(ptr %v33) #0
  br label %bb2
bb2:
  %v35 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v37 = mul i32 %v35, %v36
  %v38 = zext i32 %v24 to i64
  %v39 = zext i32 %v27 to i64
  %v40 = mul i64 %v38, %v39
  %v41 = insertvalue { i64, i64 } undef, i64 %v34, 0
  %v42 = insertvalue { i64, i64 } %v41, i64 %v40, 1
  %v43 = zext i32 %v37 to i64
  %v44 = extractvalue { i64, i64 } %v42, 0
  %v45 = extractvalue { i64, i64 } %v42, 1
  %v46 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v44, i64 %v45, i64 %v43) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v46, ptr %v31, align 8
  br label %bb14
bb5:
  %v47 = phi i64 [ %v117, %bb13 ], [ %v103, %bb14 ]
  %v48 = phi i64 [ %v118, %bb13 ], [ %v106, %bb14 ]
  %v49 = add i64 %v108, 1
  %v50 = icmp eq i64 %v49, 0
  %v51 = select i1 %v50, i8 0, i8 1
  %v52 = insertvalue { i8, { { i64 } } } undef, i8 %v51, 0
  %v53 = insertvalue { i8, { { i64 } } } %v52, i64 %v49, 1, 0, 0
  %v54 = extractvalue { i8, { { i64 } } } %v53, 0
  %v55 = zext i8 %v54 to i64
  %v56 = icmp eq i64 %v55, 1
  %v57 = extractvalue { i8, { { i64 } } } %v53, 1
  %v58 = alloca { { i64 } }, align 8
  store { { i64 } } %v57, ptr %v58, align 8
  %v59 = load i64, ptr %v58, align 8
  %v60 = icmp ugt i64 %v48, 0
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb16, label %bb15
bb6:
  unreachable
bb7:
  %v62 = extractvalue { i64, i64 } %v116, 1
  %v63 = icmp eq i64 %v39, 0
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb9, label %bb19
bb8:
  ret void
bb9:
  %v65 = udiv i64 %v62, %v39
  %v66 = urem i64 %v62, %v39
  %v67 = zext i32 %v26 to i64
  %v68 = icmp eq i64 %v67, 0
  %v69 = xor i1 %v68, 1
  br i1 %v69, label %bb10, label %bb20
bb10:
  %v70 = udiv i64 %v65, %v67
  %v71 = urem i64 %v65, %v67
  %v72 = extractvalue { ptr, i64 } %v22, 1
  %v73 = icmp ult i64 %v70, %v72
  br i1 %v73, label %bb11, label %bb21
bb11:
  %v74 = extractvalue { ptr, i64 } %v22, 0
  %v75 = getelementptr inbounds i32, ptr %v74, i64 %v70
  %v76 = load i32, ptr %v75, align 4
  %v77 = sext i32 %v76 to i64
  %v78 = mul i64 2, %v67
  %v79 = mul i64 %v78, %v39
  %v80 = mul i64 %v77, %v79
  %v81 = mul i64 %v71, %v39
  %v82 = add i64 %v80, %v81
  %v83 = add i64 %v82, %v66
  %v84 = extractvalue { ptr, i64 } %v21, 1
  %v85 = icmp ult i64 %v83, %v84
  br i1 %v85, label %bb12, label %bb22
bb12:
  %v86 = extractvalue { ptr, i64 } %v21, 0
  %v87 = getelementptr inbounds i16, ptr %v86, i64 %v83
  %v88 = load i16, ptr %v87, align 2
  %v89 = extractvalue { ptr, i64 } %v28, 0
  %v90 = getelementptr inbounds i16, ptr %v89, i64 %v62
  store i16 %v88, ptr %v90, align 2
  %v91 = mul i64 %v67, %v39
  %v92 = add i64 %v80, %v91
  %v93 = add i64 %v92, %v81
  %v94 = add i64 %v93, %v66
  %v95 = icmp ult i64 %v94, %v84
  br i1 %v95, label %bb13, label %bb23
bb13:
  %v96 = extractvalue { ptr, i64 } %v21, 0
  %v97 = getelementptr inbounds i16, ptr %v96, i64 %v94
  %v98 = load i16, ptr %v97, align 2
  %v99 = extractvalue { ptr, i64 } %v29, 0
  %v100 = getelementptr inbounds i16, ptr %v99, i64 %v62
  store i16 %v98, ptr %v100, align 2
  br label %bb5
bb14:
  %v101 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v31, i32 0, i32 0
  %v102 = getelementptr inbounds { i64, i64 }, ptr %v101, i32 0, i32 0
  %v103 = load i64, ptr %v102, align 8
  %v104 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v31, i32 0, i32 0
  %v105 = getelementptr inbounds { i64, i64 }, ptr %v104, i32 0, i32 1
  %v106 = load i64, ptr %v105, align 8
  %v107 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v31, i32 0, i32 1
  %v108 = load i64, ptr %v107, align 8
  %v109 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v31, i32 0, i32 2
  %v110 = load i1, ptr %v109, align 1
  br label %bb5
bb15:
  %v111 = add i64 %v47, %v59
  %v112 = sub i64 %v48, 1
  %v113 = insertvalue { i64, i64 } undef, i64 1, 0
  %v114 = insertvalue { i64, i64 } %v113, i64 %v47, 1
  br label %bb17
bb16:
  %v115 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb17
bb17:
  %v116 = phi { i64, i64 } [ %v114, %bb15 ], [ %v115, %bb16 ]
  %v117 = phi i64 [ %v111, %bb15 ], [ %v47, %bb16 ]
  %v118 = phi i64 [ %v112, %bb15 ], [ %v48, %bb16 ]
  %v119 = extractvalue { i64, i64 } %v116, 0
  %v120 = bitcast i64 %v119 to i64
  %v121 = icmp eq i64 %v120, 0
  br i1 %v121, label %bb8, label %bb18
bb18:
  %v122 = icmp eq i64 %v120, 1
  br i1 %v122, label %bb7, label %bb6
bb19:
  unreachable
bb20:
  unreachable
bb21:
  unreachable
bb22:
  unreachable
bb23:
  unreachable
}

define void @sanitize_nan_bf16(ptr %v0, i64 %v1, i32 %v2) #0 {
entry:
  %v3 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v4 = insertvalue { ptr, i64 } %v3, i64 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { ptr, i64 } [ %v4, %entry ]
  %v6 = phi i32 [ %v2, %entry ]
  %v7 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v8 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v9 = mul i32 %v7, %v8
  %v10 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v11 = add i32 %v9, %v10
  %v12 = zext i32 %v11 to i64
  %v13 = zext i32 %v6 to i64
  %v14 = icmp uge i64 %v12, %v13
  %v15 = xor i1 %v14, 1
  br i1 %v15, label %bb5, label %bb4
bb4:
  br label %bb8
bb5:
  %v16 = extractvalue { ptr, i64 } %v5, 1
  %v17 = icmp ult i64 %v12, %v16
  %v18 = extractvalue { ptr, i64 } %v5, 0
  %v19 = getelementptr inbounds i16, ptr %v18, i64 %v12
  %v20 = load i16, ptr %v19, align 2
  %v21 = zext i16 %v20 to i32
  %v22 = and i32 16, 31
  %v23 = shl i32 %v21, %v22
  %v24 = bitcast i32 %v23 to float
  %v25 = fcmp une float %v24, %v24
  %v26 = xor i1 %v25, 1
  br i1 %v26, label %bb7, label %bb6
bb6:
  %v27 = extractvalue { ptr, i64 } %v5, 0
  %v28 = getelementptr inbounds i16, ptr %v27, i64 %v12
  store i16 0, ptr %v28, align 2
  br label %bb7
bb7:
  br label %bb8
bb8:
  ret void
}

define void @infers_kv_cache_write_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, i32 %v10) #0 {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v1, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v3, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v5, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v7, 1
  br label %bb0
bb0:
  %v19 = phi { ptr, i64 } [ %v12, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = phi { ptr, i64 } [ %v18, %entry ]
  %v23 = phi i32 [ %v8, %entry ]
  %v24 = phi i32 [ %v9, %entry ]
  %v25 = phi i32 [ %v10, %entry ]
  %v26 = alloca {  }, align 1
  %v27 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v29 = bitcast ptr %v26 to ptr
  %v30 = call i64 @cuda_device____internal__index_1d(ptr %v29) #0
  br label %bb2
bb2:
  %v31 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v32 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v33 = mul i32 %v31, %v32
  %v34 = zext i32 %v23 to i64
  %v35 = zext i32 %v24 to i64
  %v36 = mul i64 %v34, %v35
  %v37 = insertvalue { i64, i64 } undef, i64 %v30, 0
  %v38 = insertvalue { i64, i64 } %v37, i64 %v36, 1
  %v39 = zext i32 %v33 to i64
  %v40 = extractvalue { i64, i64 } %v38, 0
  %v41 = extractvalue { i64, i64 } %v38, 1
  %v42 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v40, i64 %v41, i64 %v39) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v42, ptr %v27, align 8
  br label %bb13
bb5:
  %v43 = phi i64 [ %v104, %bb12 ], [ %v90, %bb13 ]
  %v44 = phi i64 [ %v105, %bb12 ], [ %v93, %bb13 ]
  %v45 = add i64 %v95, 1
  %v46 = icmp eq i64 %v45, 0
  %v47 = select i1 %v46, i8 0, i8 1
  %v48 = insertvalue { i8, { { i64 } } } undef, i8 %v47, 0
  %v49 = insertvalue { i8, { { i64 } } } %v48, i64 %v45, 1, 0, 0
  %v50 = extractvalue { i8, { { i64 } } } %v49, 0
  %v51 = zext i8 %v50 to i64
  %v52 = icmp eq i64 %v51, 1
  %v53 = extractvalue { i8, { { i64 } } } %v49, 1
  %v54 = alloca { { i64 } }, align 8
  store { { i64 } } %v53, ptr %v54, align 8
  %v55 = load i64, ptr %v54, align 8
  %v56 = icmp ugt i64 %v44, 0
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb15, label %bb14
bb6:
  unreachable
bb7:
  %v58 = extractvalue { i64, i64 } %v103, 1
  %v59 = icmp eq i64 %v35, 0
  %v60 = xor i1 %v59, 1
  br i1 %v60, label %bb9, label %bb18
bb8:
  ret void
bb9:
  %v61 = udiv i64 %v58, %v35
  %v62 = urem i64 %v58, %v35
  %v63 = extractvalue { ptr, i64 } %v22, 1
  %v64 = icmp ult i64 %v61, %v63
  br i1 %v64, label %bb10, label %bb19
bb10:
  %v65 = extractvalue { ptr, i64 } %v22, 0
  %v66 = getelementptr inbounds i32, ptr %v65, i64 %v61
  %v67 = load i32, ptr %v66, align 4
  %v68 = sext i32 %v67 to i64
  %v69 = mul i64 %v68, %v35
  %v70 = add i64 %v69, %v62
  %v71 = extractvalue { ptr, i64 } %v19, 1
  %v72 = icmp ult i64 %v58, %v71
  br i1 %v72, label %bb11, label %bb20
bb11:
  %v73 = extractvalue { ptr, i64 } %v19, 0
  %v74 = getelementptr inbounds i16, ptr %v73, i64 %v58
  %v75 = load i16, ptr %v74, align 2
  %v76 = extractvalue { ptr, i64 } %v21, 0
  %v77 = getelementptr inbounds i16, ptr %v76, i64 %v70
  store i16 %v75, ptr %v77, align 2
  %v78 = zext i32 %v25 to i64
  %v79 = mul i64 %v78, %v35
  %v80 = add i64 %v79, %v69
  %v81 = add i64 %v80, %v62
  %v82 = extractvalue { ptr, i64 } %v20, 1
  %v83 = icmp ult i64 %v58, %v82
  br i1 %v83, label %bb12, label %bb21
bb12:
  %v84 = extractvalue { ptr, i64 } %v20, 0
  %v85 = getelementptr inbounds i16, ptr %v84, i64 %v58
  %v86 = load i16, ptr %v85, align 2
  %v87 = getelementptr inbounds i16, ptr %v76, i64 %v81
  store i16 %v86, ptr %v87, align 2
  br label %bb5
bb13:
  %v88 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 0
  %v89 = getelementptr inbounds { i64, i64 }, ptr %v88, i32 0, i32 0
  %v90 = load i64, ptr %v89, align 8
  %v91 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 0
  %v92 = getelementptr inbounds { i64, i64 }, ptr %v91, i32 0, i32 1
  %v93 = load i64, ptr %v92, align 8
  %v94 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 1
  %v95 = load i64, ptr %v94, align 8
  %v96 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 2
  %v97 = load i1, ptr %v96, align 1
  br label %bb5
bb14:
  %v98 = add i64 %v43, %v55
  %v99 = sub i64 %v44, 1
  %v100 = insertvalue { i64, i64 } undef, i64 1, 0
  %v101 = insertvalue { i64, i64 } %v100, i64 %v43, 1
  br label %bb16
bb15:
  %v102 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb16
bb16:
  %v103 = phi { i64, i64 } [ %v101, %bb14 ], [ %v102, %bb15 ]
  %v104 = phi i64 [ %v98, %bb14 ], [ %v43, %bb15 ]
  %v105 = phi i64 [ %v99, %bb14 ], [ %v44, %bb15 ]
  %v106 = extractvalue { i64, i64 } %v103, 0
  %v107 = bitcast i64 %v106 to i64
  %v108 = icmp eq i64 %v107, 0
  br i1 %v108, label %bb8, label %bb17
bb17:
  %v109 = icmp eq i64 %v107, 1
  br i1 %v109, label %bb7, label %bb6
bb18:
  unreachable
bb19:
  unreachable
bb20:
  unreachable
bb21:
  unreachable
}

define void @infers_softmax_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi { ptr, i64 } [ %v9, %entry ]
  %v12 = phi i32 [ %v4, %entry ]
  %v13 = phi i32 [ %v5, %entry ]
  %v14 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v15 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v16 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v19 = zext i32 %v18 to i64
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v21 = zext i32 %v20 to i64
  %v22 = zext i32 %v12 to i64
  br label %bb4
bb4:
  %v23 = insertvalue { i64, i64 } undef, i64 %v21, 0
  %v24 = insertvalue { i64, i64 } %v23, i64 %v22, 1
  %v25 = extractvalue { i64, i64 } %v24, 0
  %v26 = extractvalue { i64, i64 } %v24, 1
  %v27 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v25, i64 %v26, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v27, ptr %v14, align 8
  br label %bb57
bb5:
  %v28 = phi float [ %v65, %bb15 ], [ 0xFFF0000000000000, %bb57 ]
  %v29 = phi i64 [ %v209, %bb15 ], [ %v195, %bb57 ]
  %v30 = phi i64 [ %v210, %bb15 ], [ %v198, %bb57 ]
  %v31 = add i64 %v200, 1
  %v32 = icmp eq i64 %v31, 0
  %v33 = select i1 %v32, i8 0, i8 1
  %v34 = insertvalue { i8, { { i64 } } } undef, i8 %v33, 0
  %v35 = insertvalue { i8, { { i64 } } } %v34, i64 %v31, 1, 0, 0
  %v36 = extractvalue { i8, { { i64 } } } %v35, 0
  %v37 = zext i8 %v36 to i64
  %v38 = icmp eq i64 %v37, 1
  %v39 = extractvalue { i8, { { i64 } } } %v35, 1
  %v40 = alloca { { i64 } }, align 8
  store { { i64 } } %v39, ptr %v40, align 8
  %v41 = load i64, ptr %v40, align 8
  %v42 = icmp ugt i64 %v30, 0
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb59, label %bb58
bb6:
  unreachable
bb7:
  %v44 = extractvalue { i64, i64 } %v208, 1
  %v45 = mul i64 %v19, %v22
  %v46 = add i64 %v45, %v44
  %v47 = extractvalue { ptr, i64 } %v10, 1
  %v48 = icmp ult i64 %v46, %v47
  br i1 %v48, label %bb9, label %bb72
bb8:
  %v49 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_softmax_bf16, i64 %v21
  %v50 = addrspacecast ptr addrspace(3) %v49 to ptr
  store float %v28, ptr %v50, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb16
bb9:
  %v52 = extractvalue { ptr, i64 } %v10, 0
  %v53 = getelementptr inbounds i16, ptr %v52, i64 %v46
  %v54 = load i16, ptr %v53, align 2
  %v55 = zext i16 %v54 to i32
  %v56 = and i32 16, 31
  %v57 = shl i32 %v55, %v56
  %v58 = bitcast i32 %v57 to float
  %v59 = icmp eq i32 %v13, 0
  br i1 %v59, label %bb11, label %bb10
bb10:
  %v60 = icmp ule i64 %v44, %v19
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb15, label %bb11
bb11:
  %v62 = fcmp ogt float %v58, %v28
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb13, label %bb12
bb12:
  br label %bb14
bb13:
  br label %bb14
bb14:
  %v64 = phi float [ %v58, %bb12 ], [ %v28, %bb13 ]
  br label %bb15
bb15:
  %v65 = phi float [ %v28, %bb10 ], [ %v64, %bb14 ]
  br label %bb5
bb16:
  %v66 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb17
bb17:
  %v67 = zext i32 %v66 to i64
  %v68 = udiv i64 %v67, 2
  br label %bb18
bb18:
  %v69 = phi i64 [ %v68, %bb17 ], [ %v88, %bb28 ]
  %v70 = icmp ugt i64 %v69, 0
  %v71 = xor i1 %v70, 1
  br i1 %v71, label %bb29, label %bb19
bb19:
  %v72 = icmp ult i64 %v21, %v69
  %v73 = xor i1 %v72, 1
  br i1 %v73, label %bb26, label %bb20
bb20:
  %v74 = add i64 %v21, %v69
  %v75 = icmp ult i64 %v74, %v67
  %v76 = xor i1 %v75, 1
  br i1 %v76, label %bb25, label %bb21
bb21:
  %v77 = load float, ptr %v50, align 4
  %v78 = add i64 %v21, %v69
  %v79 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_softmax_bf16, i64 %v78
  %v80 = addrspacecast ptr addrspace(3) %v79 to ptr
  %v81 = load float, ptr %v80, align 4
  %v82 = fcmp ogt float %v81, %v77
  %v83 = xor i1 %v82, 1
  br i1 %v83, label %bb23, label %bb22
bb22:
  br label %bb24
bb23:
  br label %bb24
bb24:
  %v84 = phi float [ %v81, %bb22 ], [ %v77, %bb23 ]
  store float %v84, ptr %v50, align 4
  br label %bb27
bb25:
  br label %bb27
bb26:
  br label %bb27
bb27:
  call void @llvm.nvvm.barrier0() #0
  br label %bb28
bb28:
  %v86 = zext i32 1 to i64
  %v87 = and i64 %v86, 63
  %v88 = lshr i64 %v69, %v87
  br label %bb18
bb29:
  %v89 = load float, ptr addrspace(3) @__dynamic_smem_infers_softmax_bf16, align 4
  %v90 = extractvalue { i64, i64 } %v24, 0
  %v91 = extractvalue { i64, i64 } %v24, 1
  %v92 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v90, i64 %v91, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v92, ptr %v15, align 8
  br label %bb62
bb30:
  %v93 = phi float [ %v128, %bb37 ], [ 0.0, %bb62 ]
  %v94 = phi i64 [ %v231, %bb37 ], [ %v217, %bb62 ]
  %v95 = phi i64 [ %v232, %bb37 ], [ %v220, %bb62 ]
  %v96 = add i64 %v222, 1
  %v97 = icmp eq i64 %v96, 0
  %v98 = select i1 %v97, i8 0, i8 1
  %v99 = insertvalue { i8, { { i64 } } } undef, i8 %v98, 0
  %v100 = insertvalue { i8, { { i64 } } } %v99, i64 %v96, 1, 0, 0
  %v101 = extractvalue { i8, { { i64 } } } %v100, 0
  %v102 = zext i8 %v101 to i64
  %v103 = icmp eq i64 %v102, 1
  %v104 = extractvalue { i8, { { i64 } } } %v100, 1
  %v105 = alloca { { i64 } }, align 8
  store { { i64 } } %v104, ptr %v105, align 8
  %v106 = load i64, ptr %v105, align 8
  %v107 = icmp ugt i64 %v95, 0
  %v108 = xor i1 %v107, 1
  br i1 %v108, label %bb64, label %bb63
bb31:
  %v109 = extractvalue { i64, i64 } %v230, 1
  %v110 = icmp eq i32 %v13, 0
  br i1 %v110, label %bb34, label %bb33
bb32:
  store float %v93, ptr %v50, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb38
bb33:
  %v112 = icmp ule i64 %v109, %v19
  %v113 = xor i1 %v112, 1
  br i1 %v113, label %bb37, label %bb34
bb34:
  %v114 = mul i64 %v19, %v22
  %v115 = add i64 %v114, %v109
  %v116 = extractvalue { ptr, i64 } %v10, 1
  %v117 = icmp ult i64 %v115, %v116
  br i1 %v117, label %bb35, label %bb73
bb35:
  %v118 = extractvalue { ptr, i64 } %v10, 0
  %v119 = getelementptr inbounds i16, ptr %v118, i64 %v115
  %v120 = load i16, ptr %v119, align 2
  %v121 = zext i16 %v120 to i32
  %v122 = and i32 16, 31
  %v123 = shl i32 %v121, %v122
  %v124 = bitcast i32 %v123 to float
  %v125 = fsub contract float %v124, %v89
  %v126 = call float @__nv_expf(float %v125) #0
  br label %bb36
bb36:
  %v127 = fadd contract float %v93, %v126
  br label %bb37
bb37:
  %v128 = phi float [ %v93, %bb33 ], [ %v127, %bb36 ]
  br label %bb30
bb38:
  %v129 = udiv i64 %v67, 2
  br label %bb39
bb39:
  %v130 = phi i64 [ %v129, %bb38 ], [ %v147, %bb46 ]
  %v131 = icmp ugt i64 %v130, 0
  %v132 = xor i1 %v131, 1
  br i1 %v132, label %bb47, label %bb40
bb40:
  %v133 = icmp ult i64 %v21, %v130
  %v134 = xor i1 %v133, 1
  br i1 %v134, label %bb44, label %bb41
bb41:
  %v135 = add i64 %v21, %v130
  %v136 = icmp ult i64 %v135, %v67
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb43, label %bb42
bb42:
  %v138 = load float, ptr %v50, align 4
  %v139 = add i64 %v21, %v130
  %v140 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_softmax_bf16, i64 %v139
  %v141 = addrspacecast ptr addrspace(3) %v140 to ptr
  %v142 = load float, ptr %v141, align 4
  %v143 = fadd contract float %v138, %v142
  store float %v143, ptr %v50, align 4
  br label %bb45
bb43:
  br label %bb45
bb44:
  br label %bb45
bb45:
  call void @llvm.nvvm.barrier0() #0
  br label %bb46
bb46:
  %v145 = zext i32 1 to i64
  %v146 = and i64 %v145, 63
  %v147 = lshr i64 %v130, %v146
  br label %bb39
bb47:
  %v148 = load float, ptr addrspace(3) @__dynamic_smem_infers_softmax_bf16, align 4
  %v149 = extractvalue { i64, i64 } %v24, 0
  %v150 = extractvalue { i64, i64 } %v24, 1
  %v151 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v149, i64 %v150, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v151, ptr %v16, align 8
  br label %bb67
bb48:
  %v152 = phi i64 [ %v253, %bb56 ], [ %v239, %bb67 ]
  %v153 = phi i64 [ %v254, %bb56 ], [ %v242, %bb67 ]
  %v154 = add i64 %v244, 1
  %v155 = icmp eq i64 %v154, 0
  %v156 = select i1 %v155, i8 0, i8 1
  %v157 = insertvalue { i8, { { i64 } } } undef, i8 %v156, 0
  %v158 = insertvalue { i8, { { i64 } } } %v157, i64 %v154, 1, 0, 0
  %v159 = extractvalue { i8, { { i64 } } } %v158, 0
  %v160 = zext i8 %v159 to i64
  %v161 = icmp eq i64 %v160, 1
  %v162 = extractvalue { i8, { { i64 } } } %v158, 1
  %v163 = alloca { { i64 } }, align 8
  store { { i64 } } %v162, ptr %v163, align 8
  %v164 = load i64, ptr %v163, align 8
  %v165 = icmp ugt i64 %v153, 0
  %v166 = xor i1 %v165, 1
  br i1 %v166, label %bb69, label %bb68
bb49:
  %v167 = extractvalue { i64, i64 } %v252, 1
  %v168 = mul i64 %v19, %v22
  %v169 = add i64 %v168, %v167
  %v170 = icmp eq i32 %v13, 0
  br i1 %v170, label %bb52, label %bb51
bb50:
  ret void
bb51:
  %v171 = icmp ule i64 %v167, %v19
  %v172 = xor i1 %v171, 1
  br i1 %v172, label %bb55, label %bb52
bb52:
  %v173 = extractvalue { ptr, i64 } %v10, 1
  %v174 = icmp ult i64 %v169, %v173
  br i1 %v174, label %bb53, label %bb74
bb53:
  %v175 = extractvalue { ptr, i64 } %v10, 0
  %v176 = getelementptr inbounds i16, ptr %v175, i64 %v169
  %v177 = load i16, ptr %v176, align 2
  %v178 = zext i16 %v177 to i32
  %v179 = and i32 16, 31
  %v180 = shl i32 %v178, %v179
  %v181 = bitcast i32 %v180 to float
  %v182 = fsub contract float %v181, %v89
  %v183 = call float @__nv_expf(float %v182) #0
  br label %bb54
bb54:
  %v184 = fdiv contract float %v183, %v148
  %v185 = bitcast float %v184 to i32
  %v186 = and i32 16, 31
  %v187 = lshr i32 %v185, %v186
  %v188 = trunc i32 %v187 to i16
  %v189 = extractvalue { ptr, i64 } %v11, 0
  %v190 = getelementptr inbounds i16, ptr %v189, i64 %v169
  store i16 %v188, ptr %v190, align 2
  br label %bb56
bb55:
  %v191 = extractvalue { ptr, i64 } %v11, 0
  %v192 = getelementptr inbounds i16, ptr %v191, i64 %v169
  store i16 0, ptr %v192, align 2
  br label %bb56
bb56:
  br label %bb48
bb57:
  %v193 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 0
  %v194 = getelementptr inbounds { i64, i64 }, ptr %v193, i32 0, i32 0
  %v195 = load i64, ptr %v194, align 8
  %v196 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 0
  %v197 = getelementptr inbounds { i64, i64 }, ptr %v196, i32 0, i32 1
  %v198 = load i64, ptr %v197, align 8
  %v199 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 1
  %v200 = load i64, ptr %v199, align 8
  %v201 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 2
  %v202 = load i1, ptr %v201, align 1
  br label %bb5
bb58:
  %v203 = add i64 %v29, %v41
  %v204 = sub i64 %v30, 1
  %v205 = insertvalue { i64, i64 } undef, i64 1, 0
  %v206 = insertvalue { i64, i64 } %v205, i64 %v29, 1
  br label %bb60
bb59:
  %v207 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb60
bb60:
  %v208 = phi { i64, i64 } [ %v206, %bb58 ], [ %v207, %bb59 ]
  %v209 = phi i64 [ %v203, %bb58 ], [ %v29, %bb59 ]
  %v210 = phi i64 [ %v204, %bb58 ], [ %v30, %bb59 ]
  %v211 = extractvalue { i64, i64 } %v208, 0
  %v212 = bitcast i64 %v211 to i64
  %v213 = icmp eq i64 %v212, 0
  br i1 %v213, label %bb8, label %bb61
bb61:
  %v214 = icmp eq i64 %v212, 1
  br i1 %v214, label %bb7, label %bb6
bb62:
  %v215 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 0
  %v216 = getelementptr inbounds { i64, i64 }, ptr %v215, i32 0, i32 0
  %v217 = load i64, ptr %v216, align 8
  %v218 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 0
  %v219 = getelementptr inbounds { i64, i64 }, ptr %v218, i32 0, i32 1
  %v220 = load i64, ptr %v219, align 8
  %v221 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 1
  %v222 = load i64, ptr %v221, align 8
  %v223 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 2
  %v224 = load i1, ptr %v223, align 1
  br label %bb30
bb63:
  %v225 = add i64 %v94, %v106
  %v226 = sub i64 %v95, 1
  %v227 = insertvalue { i64, i64 } undef, i64 1, 0
  %v228 = insertvalue { i64, i64 } %v227, i64 %v94, 1
  br label %bb65
bb64:
  %v229 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb65
bb65:
  %v230 = phi { i64, i64 } [ %v228, %bb63 ], [ %v229, %bb64 ]
  %v231 = phi i64 [ %v225, %bb63 ], [ %v94, %bb64 ]
  %v232 = phi i64 [ %v226, %bb63 ], [ %v95, %bb64 ]
  %v233 = extractvalue { i64, i64 } %v230, 0
  %v234 = bitcast i64 %v233 to i64
  %v235 = icmp eq i64 %v234, 0
  br i1 %v235, label %bb32, label %bb66
bb66:
  %v236 = icmp eq i64 %v234, 1
  br i1 %v236, label %bb31, label %bb6
bb67:
  %v237 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v16, i32 0, i32 0
  %v238 = getelementptr inbounds { i64, i64 }, ptr %v237, i32 0, i32 0
  %v239 = load i64, ptr %v238, align 8
  %v240 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v16, i32 0, i32 0
  %v241 = getelementptr inbounds { i64, i64 }, ptr %v240, i32 0, i32 1
  %v242 = load i64, ptr %v241, align 8
  %v243 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v16, i32 0, i32 1
  %v244 = load i64, ptr %v243, align 8
  %v245 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v16, i32 0, i32 2
  %v246 = load i1, ptr %v245, align 1
  br label %bb48
bb68:
  %v247 = add i64 %v152, %v164
  %v248 = sub i64 %v153, 1
  %v249 = insertvalue { i64, i64 } undef, i64 1, 0
  %v250 = insertvalue { i64, i64 } %v249, i64 %v152, 1
  br label %bb70
bb69:
  %v251 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb70
bb70:
  %v252 = phi { i64, i64 } [ %v250, %bb68 ], [ %v251, %bb69 ]
  %v253 = phi i64 [ %v247, %bb68 ], [ %v152, %bb69 ]
  %v254 = phi i64 [ %v248, %bb68 ], [ %v153, %bb69 ]
  %v255 = extractvalue { i64, i64 } %v252, 0
  %v256 = bitcast i64 %v255 to i64
  %v257 = icmp eq i64 %v256, 0
  br i1 %v257, label %bb50, label %bb71
bb71:
  %v258 = icmp eq i64 %v256, 1
  br i1 %v258, label %bb49, label %bb6
bb72:
  unreachable
bb73:
  unreachable
bb74:
  unreachable
}

define void @infers_embedding_gather_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7) #0 {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v3, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v5, 1
  br label %bb0
bb0:
  %v14 = phi { ptr, i64 } [ %v9, %entry ]
  %v15 = phi { ptr, i64 } [ %v11, %entry ]
  %v16 = phi { ptr, i64 } [ %v13, %entry ]
  %v17 = phi i32 [ %v6, %entry ]
  %v18 = phi i32 [ %v7, %entry ]
  %v19 = alloca {  }, align 1
  %v20 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v22 = bitcast ptr %v19 to ptr
  %v23 = call i64 @cuda_device____internal__index_1d(ptr %v22) #0
  br label %bb2
bb2:
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v25 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v26 = mul i32 %v24, %v25
  %v27 = zext i32 %v17 to i64
  %v28 = zext i32 %v18 to i64
  %v29 = mul i64 %v27, %v28
  %v30 = insertvalue { i64, i64 } undef, i64 %v23, 0
  %v31 = insertvalue { i64, i64 } %v30, i64 %v29, 1
  %v32 = zext i32 %v26 to i64
  %v33 = extractvalue { i64, i64 } %v31, 0
  %v34 = extractvalue { i64, i64 } %v31, 1
  %v35 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v33, i64 %v34, i64 %v32) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v35, ptr %v20, align 8
  br label %bb12
bb5:
  %v36 = phi i64 [ %v87, %bb11 ], [ %v73, %bb12 ]
  %v37 = phi i64 [ %v88, %bb11 ], [ %v76, %bb12 ]
  %v38 = add i64 %v78, 1
  %v39 = icmp eq i64 %v38, 0
  %v40 = select i1 %v39, i8 0, i8 1
  %v41 = insertvalue { i8, { { i64 } } } undef, i8 %v40, 0
  %v42 = insertvalue { i8, { { i64 } } } %v41, i64 %v38, 1, 0, 0
  %v43 = extractvalue { i8, { { i64 } } } %v42, 0
  %v44 = zext i8 %v43 to i64
  %v45 = icmp eq i64 %v44, 1
  %v46 = extractvalue { i8, { { i64 } } } %v42, 1
  %v47 = alloca { { i64 } }, align 8
  store { { i64 } } %v46, ptr %v47, align 8
  %v48 = load i64, ptr %v47, align 8
  %v49 = icmp ugt i64 %v37, 0
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb14, label %bb13
bb6:
  unreachable
bb7:
  %v51 = extractvalue { i64, i64 } %v86, 1
  %v52 = icmp eq i64 %v28, 0
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb9, label %bb17
bb8:
  ret void
bb9:
  %v54 = udiv i64 %v51, %v28
  %v55 = urem i64 %v51, %v28
  %v56 = extractvalue { ptr, i64 } %v15, 1
  %v57 = icmp ult i64 %v54, %v56
  br i1 %v57, label %bb10, label %bb18
bb10:
  %v58 = extractvalue { ptr, i64 } %v15, 0
  %v59 = getelementptr inbounds i32, ptr %v58, i64 %v54
  %v60 = load i32, ptr %v59, align 4
  %v61 = sext i32 %v60 to i64
  %v62 = mul i64 %v61, %v28
  %v63 = add i64 %v62, %v55
  %v64 = extractvalue { ptr, i64 } %v14, 1
  %v65 = icmp ult i64 %v63, %v64
  br i1 %v65, label %bb11, label %bb19
bb11:
  %v66 = extractvalue { ptr, i64 } %v14, 0
  %v67 = getelementptr inbounds i16, ptr %v66, i64 %v63
  %v68 = load i16, ptr %v67, align 2
  %v69 = extractvalue { ptr, i64 } %v16, 0
  %v70 = getelementptr inbounds i16, ptr %v69, i64 %v51
  store i16 %v68, ptr %v70, align 2
  br label %bb5
bb12:
  %v71 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 0
  %v72 = getelementptr inbounds { i64, i64 }, ptr %v71, i32 0, i32 0
  %v73 = load i64, ptr %v72, align 8
  %v74 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 0
  %v75 = getelementptr inbounds { i64, i64 }, ptr %v74, i32 0, i32 1
  %v76 = load i64, ptr %v75, align 8
  %v77 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 1
  %v78 = load i64, ptr %v77, align 8
  %v79 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 2
  %v80 = load i1, ptr %v79, align 1
  br label %bb5
bb13:
  %v81 = add i64 %v36, %v48
  %v82 = sub i64 %v37, 1
  %v83 = insertvalue { i64, i64 } undef, i64 1, 0
  %v84 = insertvalue { i64, i64 } %v83, i64 %v36, 1
  br label %bb15
bb14:
  %v85 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb15
bb15:
  %v86 = phi { i64, i64 } [ %v84, %bb13 ], [ %v85, %bb14 ]
  %v87 = phi i64 [ %v81, %bb13 ], [ %v36, %bb14 ]
  %v88 = phi i64 [ %v82, %bb13 ], [ %v37, %bb14 ]
  %v89 = extractvalue { i64, i64 } %v86, 0
  %v90 = bitcast i64 %v89 to i64
  %v91 = icmp eq i64 %v90, 0
  br i1 %v91, label %bb8, label %bb16
bb16:
  %v92 = icmp eq i64 %v90, 1
  br i1 %v92, label %bb7, label %bb6
bb17:
  unreachable
bb18:
  unreachable
bb19:
  unreachable
}

declare i32 @llvm.fptosi.sat.i32.f32(float)

define void @infers_argmax_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, i32 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi { ptr, i64 } [ %v9, %entry ]
  %v12 = phi i32 [ %v4, %entry ]
  %v13 = phi i32 [ %v5, %entry ]
  %v14 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v18 = zext i32 %v13 to i64
  %v19 = zext i32 %v17 to i64
  %v20 = insertvalue { i64, i64 } undef, i64 %v19, 0
  %v21 = insertvalue { i64, i64 } %v20, i64 %v18, 1
  %v22 = extractvalue { i64, i64 } %v21, 0
  %v23 = extractvalue { i64, i64 } %v21, 1
  %v24 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v22, i64 %v23, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v24, ptr %v14, align 8
  br label %bb32
bb4:
  %v25 = phi float [ %v59, %bb11 ], [ 0xFFF0000000000000, %bb32 ]
  %v26 = phi float [ %v60, %bb11 ], [ -1.0, %bb32 ]
  %v27 = phi i64 [ %v112, %bb11 ], [ %v98, %bb32 ]
  %v28 = phi i64 [ %v113, %bb11 ], [ %v101, %bb32 ]
  %v29 = add i64 %v103, 1
  %v30 = icmp eq i64 %v29, 0
  %v31 = select i1 %v30, i8 0, i8 1
  %v32 = insertvalue { i8, { { i64 } } } undef, i8 %v31, 0
  %v33 = insertvalue { i8, { { i64 } } } %v32, i64 %v29, 1, 0, 0
  %v34 = extractvalue { i8, { { i64 } } } %v33, 0
  %v35 = zext i8 %v34 to i64
  %v36 = icmp eq i64 %v35, 1
  %v37 = extractvalue { i8, { { i64 } } } %v33, 1
  %v38 = alloca { { i64 } }, align 8
  store { { i64 } } %v37, ptr %v38, align 8
  %v39 = load i64, ptr %v38, align 8
  %v40 = icmp ugt i64 %v28, 0
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb34, label %bb33
bb5:
  unreachable
bb6:
  %v42 = extractvalue { i64, i64 } %v111, 1
  %v43 = zext i32 %v16 to i64
  %v44 = mul i64 %v43, %v18
  %v45 = add i64 %v44, %v42
  %v46 = extractvalue { ptr, i64 } %v10, 1
  %v47 = icmp ult i64 %v45, %v46
  br i1 %v47, label %bb8, label %bb37
bb7:
  %v48 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v19
  br label %bb12
bb8:
  %v49 = extractvalue { ptr, i64 } %v10, 0
  %v50 = getelementptr inbounds i16, ptr %v49, i64 %v45
  %v51 = load i16, ptr %v50, align 2
  %v52 = zext i16 %v51 to i32
  %v53 = and i32 16, 31
  %v54 = shl i32 %v52, %v53
  %v55 = bitcast i32 %v54 to float
  %v56 = fcmp ogt float %v55, %v25
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb10, label %bb9
bb9:
  %v58 = uitofp i64 %v42 to float
  br label %bb11
bb10:
  br label %bb11
bb11:
  %v59 = phi float [ %v55, %bb9 ], [ %v25, %bb10 ]
  %v60 = phi float [ %v58, %bb9 ], [ %v26, %bb10 ]
  br label %bb4
bb12:
  store float %v25, ptr addrspace(3) %v48, align 4
  %v61 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v19
  br label %bb13
bb13:
  store float %v26, ptr addrspace(3) %v61, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb14
bb14:
  br label %bb15
bb15:
  %v63 = phi i32 [ 128, %bb14 ], [ %v87, %bb27 ]
  %v64 = icmp ugt i32 %v63, 0
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb28, label %bb16
bb16:
  %v66 = icmp ult i32 %v17, %v63
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb25, label %bb17
bb17:
  %v68 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v69 = add i32 %v17, %v63
  %v70 = zext i32 %v69 to i64
  %v71 = getelementptr inbounds float, ptr addrspace(3) %v68, i64 %v70
  br label %bb18
bb18:
  %v72 = load float, ptr addrspace(3) %v71, align 4
  %v73 = bitcast ptr addrspace(3) @__shared_mem_0 to ptr addrspace(3)
  %v74 = getelementptr inbounds float, ptr addrspace(3) %v73, i64 %v19
  br label %bb19
bb19:
  %v75 = load float, ptr addrspace(3) %v74, align 4
  %v76 = fcmp ogt float %v72, %v75
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb24, label %bb20
bb20:
  %v78 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_0, i64 %v19
  br label %bb21
bb21:
  store float %v72, ptr addrspace(3) %v78, align 4
  %v79 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v80 = add i32 %v17, %v63
  %v81 = zext i32 %v80 to i64
  %v82 = getelementptr inbounds float, ptr addrspace(3) %v79, i64 %v81
  br label %bb22
bb22:
  %v83 = load float, ptr addrspace(3) %v82, align 4
  %v84 = getelementptr inbounds float, ptr addrspace(3) @__shared_mem_1, i64 %v19
  br label %bb23
bb23:
  store float %v83, ptr addrspace(3) %v84, align 4
  br label %bb24
bb24:
  br label %bb26
bb25:
  br label %bb26
bb26:
  call void @llvm.nvvm.barrier0() #0
  br label %bb27
bb27:
  %v86 = and i32 1, 31
  %v87 = lshr i32 %v63, %v86
  br label %bb15
bb28:
  %v88 = icmp eq i32 %v17, 0
  br i1 %v88, label %bb29, label %bb31
bb29:
  %v89 = bitcast ptr addrspace(3) @__shared_mem_1 to ptr addrspace(3)
  %v90 = getelementptr inbounds float, ptr addrspace(3) %v89, i64 0
  br label %bb30
bb30:
  %v91 = load float, ptr addrspace(3) %v90, align 4
  %v92 = zext i32 %v16 to i64
  %v93 = extractvalue { ptr, i64 } %v11, 0
  %v94 = getelementptr inbounds i32, ptr %v93, i64 %v92
  %v95 = call i32 @llvm.fptosi.sat.i32.f32(float %v91) #0
  store i32 %v95, ptr %v94, align 4
  br label %bb31
bb31:
  ret void
bb32:
  %v96 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 0
  %v97 = getelementptr inbounds { i64, i64 }, ptr %v96, i32 0, i32 0
  %v98 = load i64, ptr %v97, align 8
  %v99 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 0
  %v100 = getelementptr inbounds { i64, i64 }, ptr %v99, i32 0, i32 1
  %v101 = load i64, ptr %v100, align 8
  %v102 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 1
  %v103 = load i64, ptr %v102, align 8
  %v104 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 2
  %v105 = load i1, ptr %v104, align 1
  br label %bb4
bb33:
  %v106 = add i64 %v27, %v39
  %v107 = sub i64 %v28, 1
  %v108 = insertvalue { i64, i64 } undef, i64 1, 0
  %v109 = insertvalue { i64, i64 } %v108, i64 %v27, 1
  br label %bb35
bb34:
  %v110 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb35
bb35:
  %v111 = phi { i64, i64 } [ %v109, %bb33 ], [ %v110, %bb34 ]
  %v112 = phi i64 [ %v106, %bb33 ], [ %v27, %bb34 ]
  %v113 = phi i64 [ %v107, %bb33 ], [ %v28, %bb34 ]
  %v114 = extractvalue { i64, i64 } %v111, 0
  %v115 = bitcast i64 %v114 to i64
  %v116 = icmp eq i64 %v115, 0
  br i1 %v116, label %bb7, label %bb36
bb36:
  %v117 = icmp eq i64 %v115, 1
  br i1 %v117, label %bb6, label %bb5
bb37:
  unreachable
}

define void @infers_add_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi { ptr, i64 } [ %v12, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = alloca {  }, align 1
  %v18 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v20 = bitcast ptr %v17 to ptr
  %v21 = call i64 @cuda_device____internal__index_1d(ptr %v20) #0
  br label %bb2
bb2:
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v24 = mul i32 %v22, %v23
  %v25 = zext i32 %v16 to i64
  %v26 = insertvalue { i64, i64 } undef, i64 %v21, 0
  %v27 = insertvalue { i64, i64 } %v26, i64 %v25, 1
  %v28 = zext i32 %v24 to i64
  %v29 = extractvalue { i64, i64 } %v27, 0
  %v30 = extractvalue { i64, i64 } %v27, 1
  %v31 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v29, i64 %v30, i64 %v28) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v31, ptr %v18, align 8
  br label %bb11
bb5:
  %v32 = phi i64 [ %v89, %bb10 ], [ %v75, %bb11 ]
  %v33 = phi i64 [ %v90, %bb10 ], [ %v78, %bb11 ]
  %v34 = add i64 %v80, 1
  %v35 = icmp eq i64 %v34, 0
  %v36 = select i1 %v35, i8 0, i8 1
  %v37 = insertvalue { i8, { { i64 } } } undef, i8 %v36, 0
  %v38 = insertvalue { i8, { { i64 } } } %v37, i64 %v34, 1, 0, 0
  %v39 = extractvalue { i8, { { i64 } } } %v38, 0
  %v40 = zext i8 %v39 to i64
  %v41 = icmp eq i64 %v40, 1
  %v42 = extractvalue { i8, { { i64 } } } %v38, 1
  %v43 = alloca { { i64 } }, align 8
  store { { i64 } } %v42, ptr %v43, align 8
  %v44 = load i64, ptr %v43, align 8
  %v45 = icmp ugt i64 %v33, 0
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb13, label %bb12
bb6:
  unreachable
bb7:
  %v47 = extractvalue { i64, i64 } %v88, 1
  %v48 = extractvalue { ptr, i64 } %v13, 1
  %v49 = icmp ult i64 %v47, %v48
  br i1 %v49, label %bb9, label %bb16
bb8:
  ret void
bb9:
  %v50 = extractvalue { ptr, i64 } %v13, 0
  %v51 = getelementptr inbounds i16, ptr %v50, i64 %v47
  %v52 = load i16, ptr %v51, align 2
  %v53 = zext i16 %v52 to i32
  %v54 = and i32 16, 31
  %v55 = shl i32 %v53, %v54
  %v56 = bitcast i32 %v55 to float
  %v57 = extractvalue { ptr, i64 } %v14, 1
  %v58 = icmp ult i64 %v47, %v57
  br i1 %v58, label %bb10, label %bb17
bb10:
  %v59 = extractvalue { ptr, i64 } %v14, 0
  %v60 = getelementptr inbounds i16, ptr %v59, i64 %v47
  %v61 = load i16, ptr %v60, align 2
  %v62 = zext i16 %v61 to i32
  %v63 = and i32 16, 31
  %v64 = shl i32 %v62, %v63
  %v65 = bitcast i32 %v64 to float
  %v66 = fadd contract float %v56, %v65
  %v67 = bitcast float %v66 to i32
  %v68 = and i32 16, 31
  %v69 = lshr i32 %v67, %v68
  %v70 = trunc i32 %v69 to i16
  %v71 = extractvalue { ptr, i64 } %v15, 0
  %v72 = getelementptr inbounds i16, ptr %v71, i64 %v47
  store i16 %v70, ptr %v72, align 2
  br label %bb5
bb11:
  %v73 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 0
  %v74 = getelementptr inbounds { i64, i64 }, ptr %v73, i32 0, i32 0
  %v75 = load i64, ptr %v74, align 8
  %v76 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 0
  %v77 = getelementptr inbounds { i64, i64 }, ptr %v76, i32 0, i32 1
  %v78 = load i64, ptr %v77, align 8
  %v79 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 1
  %v80 = load i64, ptr %v79, align 8
  %v81 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 2
  %v82 = load i1, ptr %v81, align 1
  br label %bb5
bb12:
  %v83 = add i64 %v32, %v44
  %v84 = sub i64 %v33, 1
  %v85 = insertvalue { i64, i64 } undef, i64 1, 0
  %v86 = insertvalue { i64, i64 } %v85, i64 %v32, 1
  br label %bb14
bb13:
  %v87 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb14
bb14:
  %v88 = phi { i64, i64 } [ %v86, %bb12 ], [ %v87, %bb13 ]
  %v89 = phi i64 [ %v83, %bb12 ], [ %v32, %bb13 ]
  %v90 = phi i64 [ %v84, %bb12 ], [ %v33, %bb13 ]
  %v91 = extractvalue { i64, i64 } %v88, 0
  %v92 = bitcast i64 %v91 to i64
  %v93 = icmp eq i64 %v92, 0
  br i1 %v93, label %bb8, label %bb15
bb15:
  %v94 = icmp eq i64 %v92, 1
  br i1 %v94, label %bb7, label %bb6
bb16:
  unreachable
bb17:
  unreachable
}

define void @infers_attn_output_gate_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi { ptr, i64 } [ %v12, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = alloca {  }, align 1
  %v18 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v20 = bitcast ptr %v17 to ptr
  %v21 = call i64 @cuda_device____internal__index_1d(ptr %v20) #0
  br label %bb2
bb2:
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v24 = mul i32 %v22, %v23
  %v25 = zext i32 %v16 to i64
  %v26 = insertvalue { i64, i64 } undef, i64 %v21, 0
  %v27 = insertvalue { i64, i64 } %v26, i64 %v25, 1
  %v28 = zext i32 %v24 to i64
  %v29 = extractvalue { i64, i64 } %v27, 0
  %v30 = extractvalue { i64, i64 } %v27, 1
  %v31 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v29, i64 %v30, i64 %v28) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v31, ptr %v18, align 8
  br label %bb12
bb5:
  %v32 = phi i64 [ %v93, %bb11 ], [ %v79, %bb12 ]
  %v33 = phi i64 [ %v94, %bb11 ], [ %v82, %bb12 ]
  %v34 = add i64 %v84, 1
  %v35 = icmp eq i64 %v34, 0
  %v36 = select i1 %v35, i8 0, i8 1
  %v37 = insertvalue { i8, { { i64 } } } undef, i8 %v36, 0
  %v38 = insertvalue { i8, { { i64 } } } %v37, i64 %v34, 1, 0, 0
  %v39 = extractvalue { i8, { { i64 } } } %v38, 0
  %v40 = zext i8 %v39 to i64
  %v41 = icmp eq i64 %v40, 1
  %v42 = extractvalue { i8, { { i64 } } } %v38, 1
  %v43 = alloca { { i64 } }, align 8
  store { { i64 } } %v42, ptr %v43, align 8
  %v44 = load i64, ptr %v43, align 8
  %v45 = icmp ugt i64 %v33, 0
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb14, label %bb13
bb6:
  unreachable
bb7:
  %v47 = extractvalue { i64, i64 } %v92, 1
  %v48 = extractvalue { ptr, i64 } %v13, 1
  %v49 = icmp ult i64 %v47, %v48
  br i1 %v49, label %bb9, label %bb17
bb8:
  ret void
bb9:
  %v50 = extractvalue { ptr, i64 } %v13, 0
  %v51 = getelementptr inbounds i16, ptr %v50, i64 %v47
  %v52 = load i16, ptr %v51, align 2
  %v53 = zext i16 %v52 to i32
  %v54 = and i32 16, 31
  %v55 = shl i32 %v53, %v54
  %v56 = bitcast i32 %v55 to float
  %v57 = extractvalue { ptr, i64 } %v14, 1
  %v58 = icmp ult i64 %v47, %v57
  br i1 %v58, label %bb10, label %bb18
bb10:
  %v59 = extractvalue { ptr, i64 } %v14, 0
  %v60 = getelementptr inbounds i16, ptr %v59, i64 %v47
  %v61 = load i16, ptr %v60, align 2
  %v62 = zext i16 %v61 to i32
  %v63 = and i32 16, 31
  %v64 = shl i32 %v62, %v63
  %v65 = bitcast i32 %v64 to float
  %v66 = fneg float %v65
  %v67 = call float @__nv_expf(float %v66) #0
  br label %bb11
bb11:
  %v68 = fadd contract float 1.0, %v67
  %v69 = fdiv contract float 1.0, %v68
  %v70 = fmul contract float %v56, %v69
  %v71 = bitcast float %v70 to i32
  %v72 = and i32 16, 31
  %v73 = lshr i32 %v71, %v72
  %v74 = trunc i32 %v73 to i16
  %v75 = extractvalue { ptr, i64 } %v15, 0
  %v76 = getelementptr inbounds i16, ptr %v75, i64 %v47
  store i16 %v74, ptr %v76, align 2
  br label %bb5
bb12:
  %v77 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 0
  %v78 = getelementptr inbounds { i64, i64 }, ptr %v77, i32 0, i32 0
  %v79 = load i64, ptr %v78, align 8
  %v80 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 0
  %v81 = getelementptr inbounds { i64, i64 }, ptr %v80, i32 0, i32 1
  %v82 = load i64, ptr %v81, align 8
  %v83 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 1
  %v84 = load i64, ptr %v83, align 8
  %v85 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 2
  %v86 = load i1, ptr %v85, align 1
  br label %bb5
bb13:
  %v87 = add i64 %v32, %v44
  %v88 = sub i64 %v33, 1
  %v89 = insertvalue { i64, i64 } undef, i64 1, 0
  %v90 = insertvalue { i64, i64 } %v89, i64 %v32, 1
  br label %bb15
bb14:
  %v91 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb15
bb15:
  %v92 = phi { i64, i64 } [ %v90, %bb13 ], [ %v91, %bb14 ]
  %v93 = phi i64 [ %v87, %bb13 ], [ %v32, %bb14 ]
  %v94 = phi i64 [ %v88, %bb13 ], [ %v33, %bb14 ]
  %v95 = extractvalue { i64, i64 } %v92, 0
  %v96 = bitcast i64 %v95 to i64
  %v97 = icmp eq i64 %v96, 0
  br i1 %v97, label %bb8, label %bb16
bb16:
  %v98 = icmp eq i64 %v96, 1
  br i1 %v98, label %bb7, label %bb6
bb17:
  unreachable
bb18:
  unreachable
}

define void @infers_silu_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = alloca {  }, align 1
  %v13 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v15 = bitcast ptr %v12 to ptr
  %v16 = call i64 @cuda_device____internal__index_1d(ptr %v15) #0
  br label %bb2
bb2:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v19 = mul i32 %v17, %v18
  %v20 = zext i32 %v11 to i64
  %v21 = insertvalue { i64, i64 } undef, i64 %v16, 0
  %v22 = insertvalue { i64, i64 } %v21, i64 %v20, 1
  %v23 = zext i32 %v19 to i64
  %v24 = extractvalue { i64, i64 } %v22, 0
  %v25 = extractvalue { i64, i64 } %v22, 1
  %v26 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v24, i64 %v25, i64 %v23) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v26, ptr %v13, align 8
  br label %bb11
bb5:
  %v27 = phi i64 [ %v79, %bb10 ], [ %v65, %bb11 ]
  %v28 = phi i64 [ %v80, %bb10 ], [ %v68, %bb11 ]
  %v29 = add i64 %v70, 1
  %v30 = icmp eq i64 %v29, 0
  %v31 = select i1 %v30, i8 0, i8 1
  %v32 = insertvalue { i8, { { i64 } } } undef, i8 %v31, 0
  %v33 = insertvalue { i8, { { i64 } } } %v32, i64 %v29, 1, 0, 0
  %v34 = extractvalue { i8, { { i64 } } } %v33, 0
  %v35 = zext i8 %v34 to i64
  %v36 = icmp eq i64 %v35, 1
  %v37 = extractvalue { i8, { { i64 } } } %v33, 1
  %v38 = alloca { { i64 } }, align 8
  store { { i64 } } %v37, ptr %v38, align 8
  %v39 = load i64, ptr %v38, align 8
  %v40 = icmp ugt i64 %v28, 0
  %v41 = xor i1 %v40, 1
  br i1 %v41, label %bb13, label %bb12
bb6:
  unreachable
bb7:
  %v42 = extractvalue { i64, i64 } %v78, 1
  %v43 = extractvalue { ptr, i64 } %v9, 1
  %v44 = icmp ult i64 %v42, %v43
  br i1 %v44, label %bb9, label %bb16
bb8:
  ret void
bb9:
  %v45 = extractvalue { ptr, i64 } %v9, 0
  %v46 = getelementptr inbounds i16, ptr %v45, i64 %v42
  %v47 = load i16, ptr %v46, align 2
  %v48 = zext i16 %v47 to i32
  %v49 = and i32 16, 31
  %v50 = shl i32 %v48, %v49
  %v51 = bitcast i32 %v50 to float
  %v52 = fneg float %v51
  %v53 = call float @__nv_expf(float %v52) #0
  br label %bb10
bb10:
  %v54 = fadd contract float 1.0, %v53
  %v55 = fdiv contract float 1.0, %v54
  %v56 = fmul contract float %v51, %v55
  %v57 = bitcast float %v56 to i32
  %v58 = and i32 16, 31
  %v59 = lshr i32 %v57, %v58
  %v60 = trunc i32 %v59 to i16
  %v61 = extractvalue { ptr, i64 } %v10, 0
  %v62 = getelementptr inbounds i16, ptr %v61, i64 %v42
  store i16 %v60, ptr %v62, align 2
  br label %bb5
bb11:
  %v63 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 0
  %v64 = getelementptr inbounds { i64, i64 }, ptr %v63, i32 0, i32 0
  %v65 = load i64, ptr %v64, align 8
  %v66 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 0
  %v67 = getelementptr inbounds { i64, i64 }, ptr %v66, i32 0, i32 1
  %v68 = load i64, ptr %v67, align 8
  %v69 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 1
  %v70 = load i64, ptr %v69, align 8
  %v71 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 2
  %v72 = load i1, ptr %v71, align 1
  br label %bb5
bb12:
  %v73 = add i64 %v27, %v39
  %v74 = sub i64 %v28, 1
  %v75 = insertvalue { i64, i64 } undef, i64 1, 0
  %v76 = insertvalue { i64, i64 } %v75, i64 %v27, 1
  br label %bb14
bb13:
  %v77 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb14
bb14:
  %v78 = phi { i64, i64 } [ %v76, %bb12 ], [ %v77, %bb13 ]
  %v79 = phi i64 [ %v73, %bb12 ], [ %v27, %bb13 ]
  %v80 = phi i64 [ %v74, %bb12 ], [ %v28, %bb13 ]
  %v81 = extractvalue { i64, i64 } %v78, 0
  %v82 = bitcast i64 %v81 to i64
  %v83 = icmp eq i64 %v82, 0
  br i1 %v83, label %bb8, label %bb15
bb15:
  %v84 = icmp eq i64 %v82, 1
  br i1 %v84, label %bb7, label %bb6
bb16:
  unreachable
}

define void @infers_conv1d_depthwise_silu_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8, i32 %v9) #0 {
entry:
  %v10 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v1, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v3, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v5, 1
  br label %bb0
bb0:
  %v16 = phi { ptr, i64 } [ %v11, %entry ]
  %v17 = phi { ptr, i64 } [ %v13, %entry ]
  %v18 = phi { ptr, i64 } [ %v15, %entry ]
  %v19 = phi i32 [ %v6, %entry ]
  %v20 = phi i32 [ %v7, %entry ]
  %v21 = phi i32 [ %v8, %entry ]
  %v22 = phi i32 [ %v9, %entry ]
  %v23 = alloca {  }, align 1
  %v24 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v26 = bitcast ptr %v23 to ptr
  %v27 = call i64 @cuda_device____internal__index_1d(ptr %v26) #0
  br label %bb2
bb2:
  %v28 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v29 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v30 = mul i32 %v28, %v29
  %v31 = zext i32 %v19 to i64
  %v32 = zext i32 %v21 to i64
  %v33 = mul i64 %v31, %v32
  %v34 = zext i32 %v20 to i64
  %v35 = mul i64 %v33, %v34
  %v36 = insertvalue { i64, i64 } undef, i64 %v27, 0
  %v37 = insertvalue { i64, i64 } %v36, i64 %v35, 1
  %v38 = zext i32 %v30 to i64
  %v39 = extractvalue { i64, i64 } %v37, 0
  %v40 = extractvalue { i64, i64 } %v37, 1
  %v41 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v39, i64 %v40, i64 %v38) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v41, ptr %v24, align 8
  br label %bb22
bb5:
  %v42 = phi i64 [ %v138, %bb21 ], [ %v124, %bb22 ]
  %v43 = phi i64 [ %v139, %bb21 ], [ %v127, %bb22 ]
  %v44 = add i64 %v129, 1
  %v45 = icmp eq i64 %v44, 0
  %v46 = select i1 %v45, i8 0, i8 1
  %v47 = insertvalue { i8, { { i64 } } } undef, i8 %v46, 0
  %v48 = insertvalue { i8, { { i64 } } } %v47, i64 %v44, 1, 0, 0
  %v49 = extractvalue { i8, { { i64 } } } %v48, 0
  %v50 = zext i8 %v49 to i64
  %v51 = icmp eq i64 %v50, 1
  %v52 = extractvalue { i8, { { i64 } } } %v48, 1
  %v53 = alloca { { i64 } }, align 8
  store { { i64 } } %v52, ptr %v53, align 8
  %v54 = load i64, ptr %v53, align 8
  %v55 = icmp ugt i64 %v43, 0
  %v56 = xor i1 %v55, 1
  br i1 %v56, label %bb24, label %bb23
bb6:
  unreachable
bb7:
  %v57 = extractvalue { i64, i64 } %v137, 1
  %v58 = icmp eq i64 %v34, 0
  %v59 = xor i1 %v58, 1
  br i1 %v59, label %bb9, label %bb31
bb8:
  ret void
bb9:
  %v60 = urem i64 %v57, %v34
  %v61 = udiv i64 %v57, %v34
  %v62 = icmp eq i64 %v32, 0
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb10, label %bb32
bb10:
  %v64 = urem i64 %v61, %v32
  %v65 = mul i64 %v32, %v34
  %v66 = icmp eq i64 %v65, 0
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb11, label %bb33
bb11:
  %v68 = udiv i64 %v57, %v65
  %v69 = sub i32 %v22, 1
  %v70 = zext i32 %v69 to i64
  %v71 = zext i32 %v22 to i64
  br label %bb12
bb12:
  %v72 = phi float [ 0.0, %bb11 ], [ %v113, %bb20 ]
  %v73 = phi i64 [ 0, %bb11 ], [ %v149, %bb20 ]
  %v74 = icmp ult i64 %v73, %v71
  %v75 = xor i1 %v74, 1
  br i1 %v75, label %bb28, label %bb27
bb13:
  %v76 = extractvalue { i64, i64 } %v148, 1
  %v77 = add i64 %v64, %v76
  %v78 = icmp uge i64 %v77, %v70
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb20, label %bb15
bb14:
  %v80 = fneg float %v72
  %v81 = call float @__nv_expf(float %v80) #0
  br label %bb21
bb15:
  %v82 = add i64 %v32, %v70
  %v83 = icmp ult i64 %v77, %v82
  %v84 = xor i1 %v83, 1
  br i1 %v84, label %bb19, label %bb16
bb16:
  %v85 = sub i64 %v77, %v70
  %v86 = mul i64 %v68, %v32
  %v87 = mul i64 %v86, %v34
  %v88 = mul i64 %v85, %v34
  %v89 = add i64 %v87, %v88
  %v90 = add i64 %v89, %v60
  %v91 = mul i64 %v60, %v71
  %v92 = add i64 %v91, %v76
  %v93 = extractvalue { ptr, i64 } %v16, 1
  %v94 = icmp ult i64 %v90, %v93
  br i1 %v94, label %bb17, label %bb34
bb17:
  %v95 = extractvalue { ptr, i64 } %v16, 0
  %v96 = getelementptr inbounds i16, ptr %v95, i64 %v90
  %v97 = load i16, ptr %v96, align 2
  %v98 = zext i16 %v97 to i32
  %v99 = and i32 16, 31
  %v100 = shl i32 %v98, %v99
  %v101 = bitcast i32 %v100 to float
  %v102 = extractvalue { ptr, i64 } %v17, 1
  %v103 = icmp ult i64 %v92, %v102
  br i1 %v103, label %bb18, label %bb35
bb18:
  %v104 = extractvalue { ptr, i64 } %v17, 0
  %v105 = getelementptr inbounds i16, ptr %v104, i64 %v92
  %v106 = load i16, ptr %v105, align 2
  %v107 = zext i16 %v106 to i32
  %v108 = and i32 16, 31
  %v109 = shl i32 %v107, %v108
  %v110 = bitcast i32 %v109 to float
  %v111 = fmul contract float %v101, %v110
  %v112 = fadd contract float %v72, %v111
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v113 = phi float [ %v72, %bb13 ], [ %v112, %bb18 ], [ %v72, %bb19 ]
  br label %bb12
bb21:
  %v114 = fadd contract float 1.0, %v81
  %v115 = fdiv contract float %v72, %v114
  %v116 = bitcast float %v115 to i32
  %v117 = and i32 16, 31
  %v118 = lshr i32 %v116, %v117
  %v119 = trunc i32 %v118 to i16
  %v120 = extractvalue { ptr, i64 } %v18, 0
  %v121 = getelementptr inbounds i16, ptr %v120, i64 %v57
  store i16 %v119, ptr %v121, align 2
  br label %bb5
bb22:
  %v122 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v24, i32 0, i32 0
  %v123 = getelementptr inbounds { i64, i64 }, ptr %v122, i32 0, i32 0
  %v124 = load i64, ptr %v123, align 8
  %v125 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v24, i32 0, i32 0
  %v126 = getelementptr inbounds { i64, i64 }, ptr %v125, i32 0, i32 1
  %v127 = load i64, ptr %v126, align 8
  %v128 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v24, i32 0, i32 1
  %v129 = load i64, ptr %v128, align 8
  %v130 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v24, i32 0, i32 2
  %v131 = load i1, ptr %v130, align 1
  br label %bb5
bb23:
  %v132 = add i64 %v42, %v54
  %v133 = sub i64 %v43, 1
  %v134 = insertvalue { i64, i64 } undef, i64 1, 0
  %v135 = insertvalue { i64, i64 } %v134, i64 %v42, 1
  br label %bb25
bb24:
  %v136 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb25
bb25:
  %v137 = phi { i64, i64 } [ %v135, %bb23 ], [ %v136, %bb24 ]
  %v138 = phi i64 [ %v132, %bb23 ], [ %v42, %bb24 ]
  %v139 = phi i64 [ %v133, %bb23 ], [ %v43, %bb24 ]
  %v140 = extractvalue { i64, i64 } %v137, 0
  %v141 = bitcast i64 %v140 to i64
  %v142 = icmp eq i64 %v141, 0
  br i1 %v142, label %bb8, label %bb26
bb26:
  %v143 = icmp eq i64 %v141, 1
  br i1 %v143, label %bb7, label %bb6
bb27:
  %v144 = add i64 %v73, 1
  %v145 = insertvalue { i64, i64 } undef, i64 1, 0
  %v146 = insertvalue { i64, i64 } %v145, i64 %v73, 1
  br label %bb29
bb28:
  %v147 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb29
bb29:
  %v148 = phi { i64, i64 } [ %v146, %bb27 ], [ %v147, %bb28 ]
  %v149 = phi i64 [ %v144, %bb27 ], [ %v73, %bb28 ]
  %v150 = extractvalue { i64, i64 } %v148, 0
  %v151 = bitcast i64 %v150 to i64
  %v152 = icmp eq i64 %v151, 0
  br i1 %v152, label %bb14, label %bb30
bb30:
  %v153 = icmp eq i64 %v151, 1
  br i1 %v153, label %bb13, label %bb6
bb31:
  unreachable
bb32:
  unreachable
bb33:
  unreachable
bb34:
  unreachable
bb35:
  unreachable
}

define void @infers_silu_glu_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6) #0 {
entry:
  %v7 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v1, 1
  %v9 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v3, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v5, 1
  br label %bb0
bb0:
  %v13 = phi { ptr, i64 } [ %v8, %entry ]
  %v14 = phi { ptr, i64 } [ %v10, %entry ]
  %v15 = phi { ptr, i64 } [ %v12, %entry ]
  %v16 = phi i32 [ %v6, %entry ]
  %v17 = alloca {  }, align 1
  %v18 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v20 = bitcast ptr %v17 to ptr
  %v21 = call i64 @cuda_device____internal__index_1d(ptr %v20) #0
  br label %bb2
bb2:
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb4
bb4:
  %v24 = mul i32 %v22, %v23
  %v25 = zext i32 %v16 to i64
  %v26 = insertvalue { i64, i64 } undef, i64 %v21, 0
  %v27 = insertvalue { i64, i64 } %v26, i64 %v25, 1
  %v28 = zext i32 %v24 to i64
  %v29 = extractvalue { i64, i64 } %v27, 0
  %v30 = extractvalue { i64, i64 } %v27, 1
  %v31 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v29, i64 %v30, i64 %v28) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v31, ptr %v18, align 8
  br label %bb12
bb5:
  %v32 = phi i64 [ %v94, %bb11 ], [ %v80, %bb12 ]
  %v33 = phi i64 [ %v95, %bb11 ], [ %v83, %bb12 ]
  %v34 = add i64 %v85, 1
  %v35 = icmp eq i64 %v34, 0
  %v36 = select i1 %v35, i8 0, i8 1
  %v37 = insertvalue { i8, { { i64 } } } undef, i8 %v36, 0
  %v38 = insertvalue { i8, { { i64 } } } %v37, i64 %v34, 1, 0, 0
  %v39 = extractvalue { i8, { { i64 } } } %v38, 0
  %v40 = zext i8 %v39 to i64
  %v41 = icmp eq i64 %v40, 1
  %v42 = extractvalue { i8, { { i64 } } } %v38, 1
  %v43 = alloca { { i64 } }, align 8
  store { { i64 } } %v42, ptr %v43, align 8
  %v44 = load i64, ptr %v43, align 8
  %v45 = icmp ugt i64 %v33, 0
  %v46 = xor i1 %v45, 1
  br i1 %v46, label %bb14, label %bb13
bb6:
  unreachable
bb7:
  %v47 = extractvalue { i64, i64 } %v93, 1
  %v48 = extractvalue { ptr, i64 } %v13, 1
  %v49 = icmp ult i64 %v47, %v48
  br i1 %v49, label %bb9, label %bb17
bb8:
  ret void
bb9:
  %v50 = extractvalue { ptr, i64 } %v13, 0
  %v51 = getelementptr inbounds i16, ptr %v50, i64 %v47
  %v52 = load i16, ptr %v51, align 2
  %v53 = zext i16 %v52 to i32
  %v54 = and i32 16, 31
  %v55 = shl i32 %v53, %v54
  %v56 = bitcast i32 %v55 to float
  %v57 = extractvalue { ptr, i64 } %v14, 1
  %v58 = icmp ult i64 %v47, %v57
  br i1 %v58, label %bb10, label %bb18
bb10:
  %v59 = extractvalue { ptr, i64 } %v14, 0
  %v60 = getelementptr inbounds i16, ptr %v59, i64 %v47
  %v61 = load i16, ptr %v60, align 2
  %v62 = zext i16 %v61 to i32
  %v63 = and i32 16, 31
  %v64 = shl i32 %v62, %v63
  %v65 = bitcast i32 %v64 to float
  %v66 = fneg float %v65
  %v67 = call float @__nv_expf(float %v66) #0
  br label %bb11
bb11:
  %v68 = fadd contract float 1.0, %v67
  %v69 = fdiv contract float 1.0, %v68
  %v70 = fmul contract float %v56, %v65
  %v71 = fmul contract float %v70, %v69
  %v72 = bitcast float %v71 to i32
  %v73 = and i32 16, 31
  %v74 = lshr i32 %v72, %v73
  %v75 = trunc i32 %v74 to i16
  %v76 = extractvalue { ptr, i64 } %v15, 0
  %v77 = getelementptr inbounds i16, ptr %v76, i64 %v47
  store i16 %v75, ptr %v77, align 2
  br label %bb5
bb12:
  %v78 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 0
  %v79 = getelementptr inbounds { i64, i64 }, ptr %v78, i32 0, i32 0
  %v80 = load i64, ptr %v79, align 8
  %v81 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 0
  %v82 = getelementptr inbounds { i64, i64 }, ptr %v81, i32 0, i32 1
  %v83 = load i64, ptr %v82, align 8
  %v84 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 1
  %v85 = load i64, ptr %v84, align 8
  %v86 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v18, i32 0, i32 2
  %v87 = load i1, ptr %v86, align 1
  br label %bb5
bb13:
  %v88 = add i64 %v32, %v44
  %v89 = sub i64 %v33, 1
  %v90 = insertvalue { i64, i64 } undef, i64 1, 0
  %v91 = insertvalue { i64, i64 } %v90, i64 %v32, 1
  br label %bb15
bb14:
  %v92 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb15
bb15:
  %v93 = phi { i64, i64 } [ %v91, %bb13 ], [ %v92, %bb14 ]
  %v94 = phi i64 [ %v88, %bb13 ], [ %v32, %bb14 ]
  %v95 = phi i64 [ %v89, %bb13 ], [ %v33, %bb14 ]
  %v96 = extractvalue { i64, i64 } %v93, 0
  %v97 = bitcast i64 %v96 to i64
  %v98 = icmp eq i64 %v97, 0
  br i1 %v98, label %bb8, label %bb16
bb16:
  %v99 = icmp eq i64 %v97, 1
  br i1 %v99, label %bb7, label %bb6
bb17:
  unreachable
bb18:
  unreachable
}

define void @infers_rmsnorm_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, float %v7) #0 {
entry:
  %v8 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v1, 1
  %v10 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v3, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v5, 1
  br label %bb0
bb0:
  %v14 = phi { ptr, i64 } [ %v9, %entry ]
  %v15 = phi { ptr, i64 } [ %v11, %entry ]
  %v16 = phi { ptr, i64 } [ %v13, %entry ]
  %v17 = phi i32 [ %v6, %entry ]
  %v18 = phi float [ %v7, %entry ]
  %v19 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v20 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v22 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v23 = zext i32 %v22 to i64
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v25 = zext i32 %v24 to i64
  %v26 = zext i32 %v17 to i64
  br label %bb4
bb4:
  %v27 = mul i64 %v23, %v26
  %v28 = insertvalue { i64, i64 } undef, i64 %v25, 0
  %v29 = insertvalue { i64, i64 } %v28, i64 %v26, 1
  %v30 = extractvalue { i64, i64 } %v29, 0
  %v31 = extractvalue { i64, i64 } %v29, 1
  %v32 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v30, i64 %v31, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v32, ptr %v19, align 8
  br label %bb26
bb5:
  %v33 = phi float [ %v64, %bb9 ], [ 0.0, %bb26 ]
  %v34 = phi i64 [ %v152, %bb9 ], [ %v138, %bb26 ]
  %v35 = phi i64 [ %v153, %bb9 ], [ %v141, %bb26 ]
  %v36 = add i64 %v143, 1
  %v37 = icmp eq i64 %v36, 0
  %v38 = select i1 %v37, i8 0, i8 1
  %v39 = insertvalue { i8, { { i64 } } } undef, i8 %v38, 0
  %v40 = insertvalue { i8, { { i64 } } } %v39, i64 %v36, 1, 0, 0
  %v41 = extractvalue { i8, { { i64 } } } %v40, 0
  %v42 = zext i8 %v41 to i64
  %v43 = icmp eq i64 %v42, 1
  %v44 = extractvalue { i8, { { i64 } } } %v40, 1
  %v45 = alloca { { i64 } }, align 8
  store { { i64 } } %v44, ptr %v45, align 8
  %v46 = load i64, ptr %v45, align 8
  %v47 = icmp ugt i64 %v35, 0
  %v48 = xor i1 %v47, 1
  br i1 %v48, label %bb28, label %bb27
bb6:
  unreachable
bb7:
  %v49 = extractvalue { i64, i64 } %v151, 1
  %v50 = add i64 %v27, %v49
  %v51 = extractvalue { ptr, i64 } %v14, 1
  %v52 = icmp ult i64 %v50, %v51
  br i1 %v52, label %bb9, label %bb37
bb8:
  %v53 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_rmsnorm_bf16, i64 %v25
  %v54 = addrspacecast ptr addrspace(3) %v53 to ptr
  store float %v33, ptr %v54, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb10
bb9:
  %v56 = extractvalue { ptr, i64 } %v14, 0
  %v57 = getelementptr inbounds i16, ptr %v56, i64 %v50
  %v58 = load i16, ptr %v57, align 2
  %v59 = zext i16 %v58 to i32
  %v60 = and i32 16, 31
  %v61 = shl i32 %v59, %v60
  %v62 = bitcast i32 %v61 to float
  %v63 = fmul contract float %v62, %v62
  %v64 = fadd contract float %v33, %v63
  br label %bb5
bb10:
  br label %bb11
bb11:
  %v65 = phi i32 [ 128, %bb10 ], [ %v80, %bb16 ]
  %v66 = icmp ugt i32 %v65, 0
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb17, label %bb12
bb12:
  %v68 = zext i32 %v65 to i64
  %v69 = icmp ult i64 %v25, %v68
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb14, label %bb13
bb13:
  %v71 = load float, ptr %v54, align 4
  %v72 = zext i32 %v65 to i64
  %v73 = add i64 %v25, %v72
  %v74 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_rmsnorm_bf16, i64 %v73
  %v75 = addrspacecast ptr addrspace(3) %v74 to ptr
  %v76 = load float, ptr %v75, align 4
  %v77 = fadd contract float %v71, %v76
  store float %v77, ptr %v54, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  call void @llvm.nvvm.barrier0() #0
  br label %bb16
bb16:
  %v79 = and i32 1, 31
  %v80 = lshr i32 %v65, %v79
  br label %bb11
bb17:
  %v81 = icmp eq i64 %v25, 0
  br i1 %v81, label %bb18, label %bb19
bb18:
  %v82 = load float, ptr addrspace(3) @__dynamic_smem_infers_rmsnorm_bf16, align 4
  %v83 = uitofp i64 %v26 to float
  %v84 = fdiv contract float %v82, %v83
  %v85 = fadd contract float %v84, %v18
  %v86 = call float @dev_sqrtf(float %v85) #0
  br label %bb31
bb19:
  call void @llvm.nvvm.barrier0() #0
  br label %bb20
bb20:
  %v88 = load float, ptr addrspace(3) @__dynamic_smem_infers_rmsnorm_bf16, align 4
  %v89 = extractvalue { i64, i64 } %v29, 0
  %v90 = extractvalue { i64, i64 } %v29, 1
  %v91 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v89, i64 %v90, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v91, ptr %v20, align 8
  br label %bb32
bb21:
  %v92 = phi i64 [ %v175, %bb25 ], [ %v161, %bb32 ]
  %v93 = phi i64 [ %v176, %bb25 ], [ %v164, %bb32 ]
  %v94 = add i64 %v166, 1
  %v95 = icmp eq i64 %v94, 0
  %v96 = select i1 %v95, i8 0, i8 1
  %v97 = insertvalue { i8, { { i64 } } } undef, i8 %v96, 0
  %v98 = insertvalue { i8, { { i64 } } } %v97, i64 %v94, 1, 0, 0
  %v99 = extractvalue { i8, { { i64 } } } %v98, 0
  %v100 = zext i8 %v99 to i64
  %v101 = icmp eq i64 %v100, 1
  %v102 = extractvalue { i8, { { i64 } } } %v98, 1
  %v103 = alloca { { i64 } }, align 8
  store { { i64 } } %v102, ptr %v103, align 8
  %v104 = load i64, ptr %v103, align 8
  %v105 = icmp ugt i64 %v93, 0
  %v106 = xor i1 %v105, 1
  br i1 %v106, label %bb34, label %bb33
bb22:
  %v107 = extractvalue { i64, i64 } %v174, 1
  %v108 = add i64 %v27, %v107
  %v109 = extractvalue { ptr, i64 } %v14, 1
  %v110 = icmp ult i64 %v108, %v109
  br i1 %v110, label %bb24, label %bb38
bb23:
  ret void
bb24:
  %v111 = extractvalue { ptr, i64 } %v14, 0
  %v112 = getelementptr inbounds i16, ptr %v111, i64 %v108
  %v113 = load i16, ptr %v112, align 2
  %v114 = zext i16 %v113 to i32
  %v115 = and i32 16, 31
  %v116 = shl i32 %v114, %v115
  %v117 = bitcast i32 %v116 to float
  %v118 = extractvalue { ptr, i64 } %v15, 1
  %v119 = icmp ult i64 %v107, %v118
  br i1 %v119, label %bb25, label %bb39
bb25:
  %v120 = extractvalue { ptr, i64 } %v15, 0
  %v121 = getelementptr inbounds i16, ptr %v120, i64 %v107
  %v122 = load i16, ptr %v121, align 2
  %v123 = zext i16 %v122 to i32
  %v124 = and i32 16, 31
  %v125 = shl i32 %v123, %v124
  %v126 = bitcast i32 %v125 to float
  %v127 = fmul contract float %v117, %v88
  %v128 = fadd contract float 1.0, %v126
  %v129 = fmul contract float %v127, %v128
  %v130 = bitcast float %v129 to i32
  %v131 = and i32 16, 31
  %v132 = lshr i32 %v130, %v131
  %v133 = trunc i32 %v132 to i16
  %v134 = extractvalue { ptr, i64 } %v16, 0
  %v135 = getelementptr inbounds i16, ptr %v134, i64 %v108
  store i16 %v133, ptr %v135, align 2
  br label %bb21
bb26:
  %v136 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v19, i32 0, i32 0
  %v137 = getelementptr inbounds { i64, i64 }, ptr %v136, i32 0, i32 0
  %v138 = load i64, ptr %v137, align 8
  %v139 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v19, i32 0, i32 0
  %v140 = getelementptr inbounds { i64, i64 }, ptr %v139, i32 0, i32 1
  %v141 = load i64, ptr %v140, align 8
  %v142 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v19, i32 0, i32 1
  %v143 = load i64, ptr %v142, align 8
  %v144 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v19, i32 0, i32 2
  %v145 = load i1, ptr %v144, align 1
  br label %bb5
bb27:
  %v146 = add i64 %v34, %v46
  %v147 = sub i64 %v35, 1
  %v148 = insertvalue { i64, i64 } undef, i64 1, 0
  %v149 = insertvalue { i64, i64 } %v148, i64 %v34, 1
  br label %bb29
bb28:
  %v150 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb29
bb29:
  %v151 = phi { i64, i64 } [ %v149, %bb27 ], [ %v150, %bb28 ]
  %v152 = phi i64 [ %v146, %bb27 ], [ %v34, %bb28 ]
  %v153 = phi i64 [ %v147, %bb27 ], [ %v35, %bb28 ]
  %v154 = extractvalue { i64, i64 } %v151, 0
  %v155 = bitcast i64 %v154 to i64
  %v156 = icmp eq i64 %v155, 0
  br i1 %v156, label %bb8, label %bb30
bb30:
  %v157 = icmp eq i64 %v155, 1
  br i1 %v157, label %bb7, label %bb6
bb31:
  %v158 = fdiv contract float 1.0, %v86
  store float %v158, ptr addrspace(3) @__dynamic_smem_infers_rmsnorm_bf16, align 4
  br label %bb19
bb32:
  %v159 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 0
  %v160 = getelementptr inbounds { i64, i64 }, ptr %v159, i32 0, i32 0
  %v161 = load i64, ptr %v160, align 8
  %v162 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 0
  %v163 = getelementptr inbounds { i64, i64 }, ptr %v162, i32 0, i32 1
  %v164 = load i64, ptr %v163, align 8
  %v165 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 1
  %v166 = load i64, ptr %v165, align 8
  %v167 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v20, i32 0, i32 2
  %v168 = load i1, ptr %v167, align 1
  br label %bb21
bb33:
  %v169 = add i64 %v92, %v104
  %v170 = sub i64 %v93, 1
  %v171 = insertvalue { i64, i64 } undef, i64 1, 0
  %v172 = insertvalue { i64, i64 } %v171, i64 %v92, 1
  br label %bb35
bb34:
  %v173 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb35
bb35:
  %v174 = phi { i64, i64 } [ %v172, %bb33 ], [ %v173, %bb34 ]
  %v175 = phi i64 [ %v169, %bb33 ], [ %v92, %bb34 ]
  %v176 = phi i64 [ %v170, %bb33 ], [ %v93, %bb34 ]
  %v177 = extractvalue { i64, i64 } %v174, 0
  %v178 = bitcast i64 %v177 to i64
  %v179 = icmp eq i64 %v178, 0
  br i1 %v179, label %bb23, label %bb36
bb36:
  %v180 = icmp eq i64 %v178, 1
  br i1 %v180, label %bb22, label %bb6
bb37:
  unreachable
bb38:
  unreachable
bb39:
  unreachable
}

define void @infers_l2norm_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4, float %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi { ptr, i64 } [ %v9, %entry ]
  %v12 = phi i32 [ %v4, %entry ]
  %v13 = phi float [ %v5, %entry ]
  %v14 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v15 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v18 = zext i32 %v17 to i64
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v20 = zext i32 %v19 to i64
  %v21 = zext i32 %v12 to i64
  br label %bb4
bb4:
  %v22 = mul i64 %v18, %v21
  %v23 = insertvalue { i64, i64 } undef, i64 %v20, 0
  %v24 = insertvalue { i64, i64 } %v23, i64 %v21, 1
  %v25 = extractvalue { i64, i64 } %v24, 0
  %v26 = extractvalue { i64, i64 } %v24, 1
  %v27 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v25, i64 %v26, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v27, ptr %v14, align 8
  br label %bb28
bb5:
  %v28 = phi float [ %v59, %bb9 ], [ 0.0, %bb28 ]
  %v29 = phi i64 [ %v139, %bb9 ], [ %v125, %bb28 ]
  %v30 = phi i64 [ %v140, %bb9 ], [ %v128, %bb28 ]
  %v31 = add i64 %v130, 1
  %v32 = icmp eq i64 %v31, 0
  %v33 = select i1 %v32, i8 0, i8 1
  %v34 = insertvalue { i8, { { i64 } } } undef, i8 %v33, 0
  %v35 = insertvalue { i8, { { i64 } } } %v34, i64 %v31, 1, 0, 0
  %v36 = extractvalue { i8, { { i64 } } } %v35, 0
  %v37 = zext i8 %v36 to i64
  %v38 = icmp eq i64 %v37, 1
  %v39 = extractvalue { i8, { { i64 } } } %v35, 1
  %v40 = alloca { { i64 } }, align 8
  store { { i64 } } %v39, ptr %v40, align 8
  %v41 = load i64, ptr %v40, align 8
  %v42 = icmp ugt i64 %v30, 0
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb30, label %bb29
bb6:
  unreachable
bb7:
  %v44 = extractvalue { i64, i64 } %v138, 1
  %v45 = add i64 %v22, %v44
  %v46 = extractvalue { ptr, i64 } %v10, 1
  %v47 = icmp ult i64 %v45, %v46
  br i1 %v47, label %bb9, label %bb39
bb8:
  %v48 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_l2norm_bf16, i64 %v20
  %v49 = addrspacecast ptr addrspace(3) %v48 to ptr
  store float %v28, ptr %v49, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb10
bb9:
  %v51 = extractvalue { ptr, i64 } %v10, 0
  %v52 = getelementptr inbounds i16, ptr %v51, i64 %v45
  %v53 = load i16, ptr %v52, align 2
  %v54 = zext i16 %v53 to i32
  %v55 = and i32 16, 31
  %v56 = shl i32 %v54, %v55
  %v57 = bitcast i32 %v56 to float
  %v58 = fmul contract float %v57, %v57
  %v59 = fadd contract float %v28, %v58
  br label %bb5
bb10:
  %v60 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb11
bb11:
  %v61 = zext i32 %v60 to i64
  %v62 = udiv i64 %v61, 2
  br label %bb12
bb12:
  %v63 = phi i64 [ %v62, %bb11 ], [ %v80, %bb19 ]
  %v64 = icmp ugt i64 %v63, 0
  %v65 = xor i1 %v64, 1
  br i1 %v65, label %bb20, label %bb13
bb13:
  %v66 = icmp ult i64 %v20, %v63
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb17, label %bb14
bb14:
  %v68 = add i64 %v20, %v63
  %v69 = icmp ult i64 %v68, %v61
  %v70 = xor i1 %v69, 1
  br i1 %v70, label %bb16, label %bb15
bb15:
  %v71 = load float, ptr %v49, align 4
  %v72 = add i64 %v20, %v63
  %v73 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_l2norm_bf16, i64 %v72
  %v74 = addrspacecast ptr addrspace(3) %v73 to ptr
  %v75 = load float, ptr %v74, align 4
  %v76 = fadd contract float %v71, %v75
  store float %v76, ptr %v49, align 4
  br label %bb18
bb16:
  br label %bb18
bb17:
  br label %bb18
bb18:
  call void @llvm.nvvm.barrier0() #0
  br label %bb19
bb19:
  %v78 = zext i32 1 to i64
  %v79 = and i64 %v78, 63
  %v80 = lshr i64 %v63, %v79
  br label %bb12
bb20:
  %v81 = icmp eq i64 %v20, 0
  br i1 %v81, label %bb21, label %bb22
bb21:
  %v82 = load float, ptr addrspace(3) @__dynamic_smem_infers_l2norm_bf16, align 4
  %v83 = fadd contract float %v82, %v13
  %v84 = call float @dev_sqrtf(float %v83) #0
  br label %bb33
bb22:
  call void @llvm.nvvm.barrier0() #0
  br label %bb23
bb23:
  %v86 = load float, ptr addrspace(3) @__dynamic_smem_infers_l2norm_bf16, align 4
  %v87 = extractvalue { i64, i64 } %v24, 0
  %v88 = extractvalue { i64, i64 } %v24, 1
  %v89 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v87, i64 %v88, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v89, ptr %v15, align 8
  br label %bb34
bb24:
  %v90 = phi i64 [ %v162, %bb27 ], [ %v148, %bb34 ]
  %v91 = phi i64 [ %v163, %bb27 ], [ %v151, %bb34 ]
  %v92 = add i64 %v153, 1
  %v93 = icmp eq i64 %v92, 0
  %v94 = select i1 %v93, i8 0, i8 1
  %v95 = insertvalue { i8, { { i64 } } } undef, i8 %v94, 0
  %v96 = insertvalue { i8, { { i64 } } } %v95, i64 %v92, 1, 0, 0
  %v97 = extractvalue { i8, { { i64 } } } %v96, 0
  %v98 = zext i8 %v97 to i64
  %v99 = icmp eq i64 %v98, 1
  %v100 = extractvalue { i8, { { i64 } } } %v96, 1
  %v101 = alloca { { i64 } }, align 8
  store { { i64 } } %v100, ptr %v101, align 8
  %v102 = load i64, ptr %v101, align 8
  %v103 = icmp ugt i64 %v91, 0
  %v104 = xor i1 %v103, 1
  br i1 %v104, label %bb36, label %bb35
bb25:
  %v105 = extractvalue { i64, i64 } %v161, 1
  %v106 = add i64 %v22, %v105
  %v107 = extractvalue { ptr, i64 } %v10, 1
  %v108 = icmp ult i64 %v106, %v107
  br i1 %v108, label %bb27, label %bb40
bb26:
  ret void
bb27:
  %v109 = extractvalue { ptr, i64 } %v10, 0
  %v110 = getelementptr inbounds i16, ptr %v109, i64 %v106
  %v111 = load i16, ptr %v110, align 2
  %v112 = zext i16 %v111 to i32
  %v113 = and i32 16, 31
  %v114 = shl i32 %v112, %v113
  %v115 = bitcast i32 %v114 to float
  %v116 = fmul contract float %v115, %v86
  %v117 = bitcast float %v116 to i32
  %v118 = and i32 16, 31
  %v119 = lshr i32 %v117, %v118
  %v120 = trunc i32 %v119 to i16
  %v121 = extractvalue { ptr, i64 } %v11, 0
  %v122 = getelementptr inbounds i16, ptr %v121, i64 %v106
  store i16 %v120, ptr %v122, align 2
  br label %bb24
bb28:
  %v123 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 0
  %v124 = getelementptr inbounds { i64, i64 }, ptr %v123, i32 0, i32 0
  %v125 = load i64, ptr %v124, align 8
  %v126 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 0
  %v127 = getelementptr inbounds { i64, i64 }, ptr %v126, i32 0, i32 1
  %v128 = load i64, ptr %v127, align 8
  %v129 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 1
  %v130 = load i64, ptr %v129, align 8
  %v131 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v14, i32 0, i32 2
  %v132 = load i1, ptr %v131, align 1
  br label %bb5
bb29:
  %v133 = add i64 %v29, %v41
  %v134 = sub i64 %v30, 1
  %v135 = insertvalue { i64, i64 } undef, i64 1, 0
  %v136 = insertvalue { i64, i64 } %v135, i64 %v29, 1
  br label %bb31
bb30:
  %v137 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb31
bb31:
  %v138 = phi { i64, i64 } [ %v136, %bb29 ], [ %v137, %bb30 ]
  %v139 = phi i64 [ %v133, %bb29 ], [ %v29, %bb30 ]
  %v140 = phi i64 [ %v134, %bb29 ], [ %v30, %bb30 ]
  %v141 = extractvalue { i64, i64 } %v138, 0
  %v142 = bitcast i64 %v141 to i64
  %v143 = icmp eq i64 %v142, 0
  br i1 %v143, label %bb8, label %bb32
bb32:
  %v144 = icmp eq i64 %v142, 1
  br i1 %v144, label %bb7, label %bb6
bb33:
  %v145 = fdiv contract float 1.0, %v84
  store float %v145, ptr addrspace(3) @__dynamic_smem_infers_l2norm_bf16, align 4
  br label %bb22
bb34:
  %v146 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 0
  %v147 = getelementptr inbounds { i64, i64 }, ptr %v146, i32 0, i32 0
  %v148 = load i64, ptr %v147, align 8
  %v149 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 0
  %v150 = getelementptr inbounds { i64, i64 }, ptr %v149, i32 0, i32 1
  %v151 = load i64, ptr %v150, align 8
  %v152 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 1
  %v153 = load i64, ptr %v152, align 8
  %v154 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 2
  %v155 = load i1, ptr %v154, align 1
  br label %bb24
bb35:
  %v156 = add i64 %v90, %v102
  %v157 = sub i64 %v91, 1
  %v158 = insertvalue { i64, i64 } undef, i64 1, 0
  %v159 = insertvalue { i64, i64 } %v158, i64 %v90, 1
  br label %bb37
bb36:
  %v160 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb37
bb37:
  %v161 = phi { i64, i64 } [ %v159, %bb35 ], [ %v160, %bb36 ]
  %v162 = phi i64 [ %v156, %bb35 ], [ %v90, %bb36 ]
  %v163 = phi i64 [ %v157, %bb35 ], [ %v91, %bb36 ]
  %v164 = extractvalue { i64, i64 } %v161, 0
  %v165 = bitcast i64 %v164 to i64
  %v166 = icmp eq i64 %v165, 0
  br i1 %v166, label %bb26, label %bb38
bb38:
  %v167 = icmp eq i64 %v165, 1
  br i1 %v167, label %bb25, label %bb6
bb39:
  unreachable
bb40:
  unreachable
}

define void @infers_rms_norm_gated_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, i32 %v8, i32 %v9, float %v10) #0 {
entry:
  %v11 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v1, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v3, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v5, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v7, 1
  br label %bb0
bb0:
  %v19 = phi { ptr, i64 } [ %v12, %entry ]
  %v20 = phi { ptr, i64 } [ %v14, %entry ]
  %v21 = phi { ptr, i64 } [ %v16, %entry ]
  %v22 = phi { ptr, i64 } [ %v18, %entry ]
  %v23 = phi i32 [ %v8, %entry ]
  %v24 = phi i32 [ %v9, %entry ]
  %v25 = phi float [ %v10, %entry ]
  %v26 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v27 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v29 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v30 = zext i32 %v29 to i64
  %v31 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v32 = zext i32 %v31 to i64
  %v33 = zext i32 %v24 to i64
  br label %bb4
bb4:
  %v34 = mul i64 %v30, %v33
  %v35 = insertvalue { i64, i64 } undef, i64 %v32, 0
  %v36 = insertvalue { i64, i64 } %v35, i64 %v33, 1
  %v37 = extractvalue { i64, i64 } %v36, 0
  %v38 = extractvalue { i64, i64 } %v36, 1
  %v39 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v37, i64 %v38, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v39, ptr %v26, align 8
  br label %bb28
bb5:
  %v40 = phi float [ %v71, %bb9 ], [ 0.0, %bb28 ]
  %v41 = phi i64 [ %v172, %bb9 ], [ %v158, %bb28 ]
  %v42 = phi i64 [ %v173, %bb9 ], [ %v161, %bb28 ]
  %v43 = add i64 %v163, 1
  %v44 = icmp eq i64 %v43, 0
  %v45 = select i1 %v44, i8 0, i8 1
  %v46 = insertvalue { i8, { { i64 } } } undef, i8 %v45, 0
  %v47 = insertvalue { i8, { { i64 } } } %v46, i64 %v43, 1, 0, 0
  %v48 = extractvalue { i8, { { i64 } } } %v47, 0
  %v49 = zext i8 %v48 to i64
  %v50 = icmp eq i64 %v49, 1
  %v51 = extractvalue { i8, { { i64 } } } %v47, 1
  %v52 = alloca { { i64 } }, align 8
  store { { i64 } } %v51, ptr %v52, align 8
  %v53 = load i64, ptr %v52, align 8
  %v54 = icmp ugt i64 %v42, 0
  %v55 = xor i1 %v54, 1
  br i1 %v55, label %bb30, label %bb29
bb6:
  unreachable
bb7:
  %v56 = extractvalue { i64, i64 } %v171, 1
  %v57 = add i64 %v34, %v56
  %v58 = extractvalue { ptr, i64 } %v19, 1
  %v59 = icmp ult i64 %v57, %v58
  br i1 %v59, label %bb9, label %bb39
bb8:
  %v60 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_rms_norm_gated_bf16, i64 %v32
  %v61 = addrspacecast ptr addrspace(3) %v60 to ptr
  store float %v40, ptr %v61, align 4
  call void @llvm.nvvm.barrier0() #0
  br label %bb10
bb9:
  %v63 = extractvalue { ptr, i64 } %v19, 0
  %v64 = getelementptr inbounds i16, ptr %v63, i64 %v57
  %v65 = load i16, ptr %v64, align 2
  %v66 = zext i16 %v65 to i32
  %v67 = and i32 16, 31
  %v68 = shl i32 %v66, %v67
  %v69 = bitcast i32 %v68 to float
  %v70 = fmul contract float %v69, %v69
  %v71 = fadd contract float %v40, %v70
  br label %bb5
bb10:
  %v72 = udiv i64 %v33, 2
  br label %bb11
bb11:
  %v73 = phi i64 [ %v72, %bb10 ], [ %v87, %bb16 ]
  %v74 = icmp ugt i64 %v73, 0
  %v75 = xor i1 %v74, 1
  br i1 %v75, label %bb17, label %bb12
bb12:
  %v76 = icmp ult i64 %v32, %v73
  %v77 = xor i1 %v76, 1
  br i1 %v77, label %bb14, label %bb13
bb13:
  %v78 = load float, ptr %v61, align 4
  %v79 = add i64 %v32, %v73
  %v80 = getelementptr inbounds float, ptr addrspace(3) @__dynamic_smem_infers_rms_norm_gated_bf16, i64 %v79
  %v81 = addrspacecast ptr addrspace(3) %v80 to ptr
  %v82 = load float, ptr %v81, align 4
  %v83 = fadd contract float %v78, %v82
  store float %v83, ptr %v61, align 4
  br label %bb15
bb14:
  br label %bb15
bb15:
  call void @llvm.nvvm.barrier0() #0
  br label %bb16
bb16:
  %v85 = zext i32 1 to i64
  %v86 = and i64 %v85, 63
  %v87 = lshr i64 %v73, %v86
  br label %bb11
bb17:
  %v88 = icmp eq i64 %v32, 0
  br i1 %v88, label %bb18, label %bb19
bb18:
  %v89 = load float, ptr addrspace(3) @__dynamic_smem_infers_rms_norm_gated_bf16, align 4
  %v90 = uitofp i64 %v33 to float
  %v91 = fdiv contract float %v89, %v90
  %v92 = fadd contract float %v91, %v25
  %v93 = call float @dev_sqrtf(float %v92) #0
  br label %bb33
bb19:
  call void @llvm.nvvm.barrier0() #0
  br label %bb20
bb20:
  %v95 = load float, ptr addrspace(3) @__dynamic_smem_infers_rms_norm_gated_bf16, align 4
  %v96 = extractvalue { i64, i64 } %v36, 0
  %v97 = extractvalue { i64, i64 } %v36, 1
  %v98 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v96, i64 %v97, i64 256) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v98, ptr %v27, align 8
  br label %bb34
bb21:
  %v99 = phi i64 [ %v195, %bb27 ], [ %v181, %bb34 ]
  %v100 = phi i64 [ %v196, %bb27 ], [ %v184, %bb34 ]
  %v101 = add i64 %v186, 1
  %v102 = icmp eq i64 %v101, 0
  %v103 = select i1 %v102, i8 0, i8 1
  %v104 = insertvalue { i8, { { i64 } } } undef, i8 %v103, 0
  %v105 = insertvalue { i8, { { i64 } } } %v104, i64 %v101, 1, 0, 0
  %v106 = extractvalue { i8, { { i64 } } } %v105, 0
  %v107 = zext i8 %v106 to i64
  %v108 = icmp eq i64 %v107, 1
  %v109 = extractvalue { i8, { { i64 } } } %v105, 1
  %v110 = alloca { { i64 } }, align 8
  store { { i64 } } %v109, ptr %v110, align 8
  %v111 = load i64, ptr %v110, align 8
  %v112 = icmp ugt i64 %v100, 0
  %v113 = xor i1 %v112, 1
  br i1 %v113, label %bb36, label %bb35
bb22:
  %v114 = extractvalue { i64, i64 } %v194, 1
  %v115 = add i64 %v34, %v114
  %v116 = extractvalue { ptr, i64 } %v19, 1
  %v117 = icmp ult i64 %v115, %v116
  br i1 %v117, label %bb24, label %bb40
bb23:
  ret void
bb24:
  %v118 = extractvalue { ptr, i64 } %v19, 0
  %v119 = getelementptr inbounds i16, ptr %v118, i64 %v115
  %v120 = load i16, ptr %v119, align 2
  %v121 = zext i16 %v120 to i32
  %v122 = and i32 16, 31
  %v123 = shl i32 %v121, %v122
  %v124 = bitcast i32 %v123 to float
  %v125 = extractvalue { ptr, i64 } %v20, 1
  %v126 = icmp ult i64 %v115, %v125
  br i1 %v126, label %bb25, label %bb41
bb25:
  %v127 = extractvalue { ptr, i64 } %v20, 0
  %v128 = getelementptr inbounds i16, ptr %v127, i64 %v115
  %v129 = load i16, ptr %v128, align 2
  %v130 = zext i16 %v129 to i32
  %v131 = and i32 16, 31
  %v132 = shl i32 %v130, %v131
  %v133 = bitcast i32 %v132 to float
  %v134 = extractvalue { ptr, i64 } %v21, 1
  %v135 = icmp ult i64 %v114, %v134
  br i1 %v135, label %bb26, label %bb42
bb26:
  %v136 = extractvalue { ptr, i64 } %v21, 0
  %v137 = getelementptr inbounds i16, ptr %v136, i64 %v114
  %v138 = load i16, ptr %v137, align 2
  %v139 = zext i16 %v138 to i32
  %v140 = and i32 16, 31
  %v141 = shl i32 %v139, %v140
  %v142 = bitcast i32 %v141 to float
  %v143 = fmul contract float %v124, %v95
  %v144 = fneg float %v133
  %v145 = call float @__nv_expf(float %v144) #0
  br label %bb27
bb27:
  %v146 = fadd contract float 1.0, %v145
  %v147 = fdiv contract float %v133, %v146
  %v148 = fmul contract float %v142, %v143
  %v149 = fmul contract float %v148, %v147
  %v150 = bitcast float %v149 to i32
  %v151 = and i32 16, 31
  %v152 = lshr i32 %v150, %v151
  %v153 = trunc i32 %v152 to i16
  %v154 = extractvalue { ptr, i64 } %v22, 0
  %v155 = getelementptr inbounds i16, ptr %v154, i64 %v115
  store i16 %v153, ptr %v155, align 2
  br label %bb21
bb28:
  %v156 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v26, i32 0, i32 0
  %v157 = getelementptr inbounds { i64, i64 }, ptr %v156, i32 0, i32 0
  %v158 = load i64, ptr %v157, align 8
  %v159 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v26, i32 0, i32 0
  %v160 = getelementptr inbounds { i64, i64 }, ptr %v159, i32 0, i32 1
  %v161 = load i64, ptr %v160, align 8
  %v162 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v26, i32 0, i32 1
  %v163 = load i64, ptr %v162, align 8
  %v164 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v26, i32 0, i32 2
  %v165 = load i1, ptr %v164, align 1
  br label %bb5
bb29:
  %v166 = add i64 %v41, %v53
  %v167 = sub i64 %v42, 1
  %v168 = insertvalue { i64, i64 } undef, i64 1, 0
  %v169 = insertvalue { i64, i64 } %v168, i64 %v41, 1
  br label %bb31
bb30:
  %v170 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb31
bb31:
  %v171 = phi { i64, i64 } [ %v169, %bb29 ], [ %v170, %bb30 ]
  %v172 = phi i64 [ %v166, %bb29 ], [ %v41, %bb30 ]
  %v173 = phi i64 [ %v167, %bb29 ], [ %v42, %bb30 ]
  %v174 = extractvalue { i64, i64 } %v171, 0
  %v175 = bitcast i64 %v174 to i64
  %v176 = icmp eq i64 %v175, 0
  br i1 %v176, label %bb8, label %bb32
bb32:
  %v177 = icmp eq i64 %v175, 1
  br i1 %v177, label %bb7, label %bb6
bb33:
  %v178 = fdiv contract float 1.0, %v93
  store float %v178, ptr addrspace(3) @__dynamic_smem_infers_rms_norm_gated_bf16, align 4
  br label %bb19
bb34:
  %v179 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 0
  %v180 = getelementptr inbounds { i64, i64 }, ptr %v179, i32 0, i32 0
  %v181 = load i64, ptr %v180, align 8
  %v182 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 0
  %v183 = getelementptr inbounds { i64, i64 }, ptr %v182, i32 0, i32 1
  %v184 = load i64, ptr %v183, align 8
  %v185 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 1
  %v186 = load i64, ptr %v185, align 8
  %v187 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v27, i32 0, i32 2
  %v188 = load i1, ptr %v187, align 1
  br label %bb21
bb35:
  %v189 = add i64 %v99, %v111
  %v190 = sub i64 %v100, 1
  %v191 = insertvalue { i64, i64 } undef, i64 1, 0
  %v192 = insertvalue { i64, i64 } %v191, i64 %v99, 1
  br label %bb37
bb36:
  %v193 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb37
bb37:
  %v194 = phi { i64, i64 } [ %v192, %bb35 ], [ %v193, %bb36 ]
  %v195 = phi i64 [ %v189, %bb35 ], [ %v99, %bb36 ]
  %v196 = phi i64 [ %v190, %bb35 ], [ %v100, %bb36 ]
  %v197 = extractvalue { i64, i64 } %v194, 0
  %v198 = bitcast i64 %v197 to i64
  %v199 = icmp eq i64 %v198, 0
  br i1 %v199, label %bb23, label %bb38
bb38:
  %v200 = icmp eq i64 %v198, 1
  br i1 %v200, label %bb22, label %bb6
bb39:
  unreachable
bb40:
  unreachable
bb41:
  unreachable
bb42:
  unreachable
}

define void @infers_fp8_dequantize_e4m3(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v13 = extractvalue { ptr, i64 } %v9, 0
  %v14 = extractvalue { ptr, i64 } %v9, 1
  %v15 = extractvalue { ptr, i64 } %v10, 0
  %v16 = extractvalue { ptr, i64 } %v10, 1
  call void @fp8_dequantize_innerNtB2_7Fp8E4M3EB4_(ptr %v13, i64 %v14, ptr %v15, i64 %v16, i32 %v11) #0
  br label %bb2
bb2:
  ret void
}

define void @infers_fp8_quantize_e5m2(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v13 = extractvalue { ptr, i64 } %v9, 0
  %v14 = extractvalue { ptr, i64 } %v9, 1
  %v15 = extractvalue { ptr, i64 } %v10, 0
  %v16 = extractvalue { ptr, i64 } %v10, 1
  call void @fp8_quantize_innerNtB2_7Fp8E5M2EB4_(ptr %v13, i64 %v14, ptr %v15, i64 %v16, i32 %v11) #0
  br label %bb2
bb2:
  ret void
}

define void @infers_fp8_quantize_e4m3(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v13 = extractvalue { ptr, i64 } %v9, 0
  %v14 = extractvalue { ptr, i64 } %v9, 1
  %v15 = extractvalue { ptr, i64 } %v10, 0
  %v16 = extractvalue { ptr, i64 } %v10, 1
  call void @fp8_quantize_innerNtB2_7Fp8E4M3EB4_(ptr %v13, i64 %v14, ptr %v15, i64 %v16, i32 %v11) #0
  br label %bb2
bb2:
  ret void
}

define void @infers_fp8_dequantize_e5m2(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v13 = extractvalue { ptr, i64 } %v9, 0
  %v14 = extractvalue { ptr, i64 } %v9, 1
  %v15 = extractvalue { ptr, i64 } %v10, 0
  %v16 = extractvalue { ptr, i64 } %v10, 1
  call void @fp8_dequantize_innerNtB2_7Fp8E5M2EB4_(ptr %v13, i64 %v14, ptr %v15, i64 %v16, i32 %v11) #0
  br label %bb2
bb2:
  ret void
}

define void @nvfp4_gemm_v3_ksplit(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, float %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12) #0 {
entry:
  %v13 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v5, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v7, 1
  br label %bb0
bb0:
  %v21 = phi { ptr, i64 } [ %v14, %entry ]
  %v22 = phi { ptr, i64 } [ %v16, %entry ]
  %v23 = phi { ptr, i64 } [ %v18, %entry ]
  %v24 = phi { ptr, i64 } [ %v20, %entry ]
  %v25 = phi float [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi i32 [ %v10, %entry ]
  %v28 = phi i32 [ %v11, %entry ]
  %v29 = phi i32 [ %v12, %entry ]
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v31 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v32 = mul i32 %v31, 64
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v34 = add i32 %v32, %v33
  %v35 = zext i32 %v34 to i64
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb4
bb4:
  %v37 = zext i32 %v36 to i64
  %v38 = zext i32 %v26 to i64
  %v39 = zext i32 %v27 to i64
  %v40 = zext i32 %v28 to i64
  %v41 = icmp uge i64 %v35, %v38
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb6, label %bb5
bb5:
  br label %bb69
bb6:
  %v43 = icmp eq i64 %v40, 0
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb7, label %bb106
bb7:
  %v45 = udiv i64 %v39, %v40
  %v46 = zext i32 %v29 to i64
  %v47 = add i64 %v45, %v46
  %v48 = sub i64 %v47, 1
  %v49 = icmp eq i64 %v46, 0
  %v50 = xor i1 %v49, 1
  br i1 %v50, label %bb8, label %bb107
bb8:
  %v51 = udiv i64 %v48, %v46
  %v52 = mul i64 %v37, %v51
  %v53 = icmp uge i64 %v52, %v45
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb11, label %bb9
bb9:
  %v55 = mul i64 %v37, %v38
  %v56 = add i64 %v55, %v35
  %v57 = extractvalue { ptr, i64 } %v21, 1
  %v58 = icmp ult i64 %v56, %v57
  br i1 %v58, label %bb10, label %bb108
bb10:
  %v59 = extractvalue { ptr, i64 } %v21, 0
  %v60 = getelementptr inbounds float, ptr %v59, i64 %v56
  store float 0.0, ptr %v60, align 4
  br label %bb69
bb11:
  %v61 = add i64 %v52, %v51
  %v62 = icmp ugt i64 %v61, %v45
  %v63 = xor i1 %v62, 1
  br i1 %v63, label %bb13, label %bb12
bb12:
  br label %bb14
bb13:
  br label %bb14
bb14:
  %v64 = phi i64 [ %v45, %bb12 ], [ %v61, %bb13 ]
  br label %bb15
bb15:
  %v65 = phi float [ 0.0, %bb14 ], [ %v258, %bb67 ]
  %v66 = phi float [ 0.0, %bb14 ], [ %v260, %bb67 ]
  %v67 = phi float [ 0.0, %bb14 ], [ %v425, %bb67 ]
  %v68 = phi float [ 0.0, %bb14 ], [ %v427, %bb67 ]
  %v69 = phi i64 [ %v52, %bb14 ], [ %v435, %bb67 ]
  %v70 = icmp ult i64 %v69, %v64
  %v71 = xor i1 %v70, 1
  br i1 %v71, label %bb71, label %bb70
bb16:
  unreachable
bb17:
  %v72 = extractvalue { i64, i64 } %v434, 1
  %v73 = mul i64 %v72, %v40
  %v74 = mul i64 %v35, %v45
  %v75 = add i64 %v74, %v72
  %v76 = extractvalue { ptr, i64 } %v23, 1
  %v77 = icmp ult i64 %v75, %v76
  %v78 = extractvalue { ptr, i64 } %v23, 0
  %v79 = getelementptr inbounds i8, ptr %v78, i64 %v75
  %v80 = load i8, ptr %v79, align 1
  %v81 = call float @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v80) #0
  br label %bb19
bb18:
  %v82 = fadd contract float %v65, %v66
  %v83 = fadd contract float %v82, %v67
  %v84 = fadd contract float %v83, %v68
  %v85 = mul i64 %v37, %v38
  %v86 = add i64 %v85, %v35
  %v87 = extractvalue { ptr, i64 } %v21, 1
  %v88 = icmp ult i64 %v86, %v87
  br i1 %v88, label %bb68, label %bb109
bb19:
  %v89 = fdiv contract float %v81, %v25
  %v90 = udiv i64 %v39, 2
  %v91 = mul i64 %v35, %v90
  %v92 = udiv i64 %v73, 2
  %v93 = add i64 %v91, %v92
  %v94 = extractvalue { ptr, i64 } %v22, 1
  %v95 = icmp ult i64 %v93, %v94
  %v96 = extractvalue { ptr, i64 } %v22, 0
  %v97 = getelementptr inbounds i8, ptr %v96, i64 %v93
  %v98 = load i8, ptr %v97, align 1
  %v99 = zext i8 %v98 to i32
  %v100 = add i64 %v93, 1
  %v101 = icmp ult i64 %v100, %v94
  %v102 = extractvalue { ptr, i64 } %v22, 0
  %v103 = getelementptr inbounds i8, ptr %v102, i64 %v100
  %v104 = load i8, ptr %v103, align 1
  %v105 = zext i8 %v104 to i32
  %v106 = add i64 %v93, 2
  %v107 = icmp ult i64 %v106, %v94
  %v108 = extractvalue { ptr, i64 } %v22, 0
  %v109 = getelementptr inbounds i8, ptr %v108, i64 %v106
  %v110 = load i8, ptr %v109, align 1
  %v111 = zext i8 %v110 to i32
  %v112 = add i64 %v93, 3
  %v113 = icmp ult i64 %v112, %v94
  %v114 = extractvalue { ptr, i64 } %v22, 0
  %v115 = getelementptr inbounds i8, ptr %v114, i64 %v112
  %v116 = load i8, ptr %v115, align 1
  %v117 = zext i8 %v116 to i32
  %v118 = and i32 8, 31
  %v119 = shl i32 %v105, %v118
  %v120 = or i32 %v99, %v119
  %v121 = and i32 16, 31
  %v122 = shl i32 %v111, %v121
  %v123 = or i32 %v120, %v122
  %v124 = and i32 24, 31
  %v125 = shl i32 %v117, %v124
  %v126 = or i32 %v123, %v125
  %v127 = and i32 %v126, 15
  %v128 = trunc i32 %v127 to i8
  %v129 = and i32 4, 31
  %v130 = lshr i32 %v126, %v129
  %v131 = and i32 %v130, 15
  %v132 = trunc i32 %v131 to i8
  %v133 = and i32 8, 31
  %v134 = lshr i32 %v126, %v133
  %v135 = and i32 %v134, 15
  %v136 = trunc i32 %v135 to i8
  %v137 = and i32 12, 31
  %v138 = lshr i32 %v126, %v137
  %v139 = and i32 %v138, 15
  %v140 = trunc i32 %v139 to i8
  %v141 = and i32 16, 31
  %v142 = lshr i32 %v126, %v141
  %v143 = and i32 %v142, 15
  %v144 = trunc i32 %v143 to i8
  %v145 = and i32 20, 31
  %v146 = lshr i32 %v126, %v145
  %v147 = and i32 %v146, 15
  %v148 = trunc i32 %v147 to i8
  %v149 = and i32 24, 31
  %v150 = lshr i32 %v126, %v149
  %v151 = and i32 %v150, 15
  %v152 = trunc i32 %v151 to i8
  %v153 = and i32 28, 31
  %v154 = lshr i32 %v126, %v153
  %v155 = and i32 %v154, 15
  %v156 = trunc i32 %v155 to i8
  %v157 = extractvalue { ptr, i64 } %v24, 1
  %v158 = icmp ult i64 %v73, %v157
  %v159 = extractvalue { ptr, i64 } %v24, 0
  %v160 = getelementptr inbounds i16, ptr %v159, i64 %v73
  %v161 = load i16, ptr %v160, align 2
  %v162 = zext i16 %v161 to i32
  %v163 = and i32 16, 31
  %v164 = shl i32 %v162, %v163
  %v165 = bitcast i32 %v164 to float
  %v166 = add i64 %v73, 1
  %v167 = icmp ult i64 %v166, %v157
  %v168 = extractvalue { ptr, i64 } %v24, 0
  %v169 = getelementptr inbounds i16, ptr %v168, i64 %v166
  %v170 = load i16, ptr %v169, align 2
  %v171 = zext i16 %v170 to i32
  %v172 = and i32 16, 31
  %v173 = shl i32 %v171, %v172
  %v174 = bitcast i32 %v173 to float
  %v175 = add i64 %v73, 2
  %v176 = icmp ult i64 %v175, %v157
  %v177 = extractvalue { ptr, i64 } %v24, 0
  %v178 = getelementptr inbounds i16, ptr %v177, i64 %v175
  %v179 = load i16, ptr %v178, align 2
  %v180 = zext i16 %v179 to i32
  %v181 = and i32 16, 31
  %v182 = shl i32 %v180, %v181
  %v183 = bitcast i32 %v182 to float
  %v184 = add i64 %v73, 3
  %v185 = icmp ult i64 %v184, %v157
  %v186 = extractvalue { ptr, i64 } %v24, 0
  %v187 = getelementptr inbounds i16, ptr %v186, i64 %v184
  %v188 = load i16, ptr %v187, align 2
  %v189 = zext i16 %v188 to i32
  %v190 = and i32 16, 31
  %v191 = shl i32 %v189, %v190
  %v192 = bitcast i32 %v191 to float
  %v193 = add i64 %v73, 4
  %v194 = icmp ult i64 %v193, %v157
  %v195 = extractvalue { ptr, i64 } %v24, 0
  %v196 = getelementptr inbounds i16, ptr %v195, i64 %v193
  %v197 = load i16, ptr %v196, align 2
  %v198 = zext i16 %v197 to i32
  %v199 = and i32 16, 31
  %v200 = shl i32 %v198, %v199
  %v201 = bitcast i32 %v200 to float
  %v202 = add i64 %v73, 5
  %v203 = icmp ult i64 %v202, %v157
  %v204 = extractvalue { ptr, i64 } %v24, 0
  %v205 = getelementptr inbounds i16, ptr %v204, i64 %v202
  %v206 = load i16, ptr %v205, align 2
  %v207 = zext i16 %v206 to i32
  %v208 = and i32 16, 31
  %v209 = shl i32 %v207, %v208
  %v210 = bitcast i32 %v209 to float
  %v211 = add i64 %v73, 6
  %v212 = icmp ult i64 %v211, %v157
  %v213 = extractvalue { ptr, i64 } %v24, 0
  %v214 = getelementptr inbounds i16, ptr %v213, i64 %v211
  %v215 = load i16, ptr %v214, align 2
  %v216 = zext i16 %v215 to i32
  %v217 = and i32 16, 31
  %v218 = shl i32 %v216, %v217
  %v219 = bitcast i32 %v218 to float
  %v220 = add i64 %v73, 7
  %v221 = icmp ult i64 %v220, %v157
  %v222 = extractvalue { ptr, i64 } %v24, 0
  %v223 = getelementptr inbounds i16, ptr %v222, i64 %v220
  %v224 = load i16, ptr %v223, align 2
  %v225 = zext i16 %v224 to i32
  %v226 = and i32 16, 31
  %v227 = shl i32 %v225, %v226
  %v228 = bitcast i32 %v227 to float
  %v229 = call float @fp4_e2m1_to_f32(i8 %v128) #0
  br label %bb74
bb20:
  br label %bb22
bb21:
  br label %bb22
bb22:
  %v230 = phi float [ %v448, %bb20 ], [ 0.0, %bb21 ]
  %v231 = call float @fp4_e2m1_to_f32(i8 %v132) #0
  br label %bb76
bb23:
  br label %bb25
bb24:
  br label %bb25
bb25:
  %v232 = phi float [ %v460, %bb23 ], [ 0.0, %bb24 ]
  %v233 = call float @fp4_e2m1_to_f32(i8 %v136) #0
  br label %bb78
bb26:
  br label %bb28
bb27:
  br label %bb28
bb28:
  %v234 = phi float [ %v472, %bb26 ], [ 0.0, %bb27 ]
  %v235 = call float @fp4_e2m1_to_f32(i8 %v140) #0
  br label %bb80
bb29:
  br label %bb31
bb30:
  br label %bb31
bb31:
  %v236 = phi float [ %v484, %bb29 ], [ 0.0, %bb30 ]
  %v237 = call float @fp4_e2m1_to_f32(i8 %v144) #0
  br label %bb82
bb32:
  br label %bb34
bb33:
  br label %bb34
bb34:
  %v238 = phi float [ %v496, %bb32 ], [ 0.0, %bb33 ]
  %v239 = call float @fp4_e2m1_to_f32(i8 %v148) #0
  br label %bb84
bb35:
  br label %bb37
bb36:
  br label %bb37
bb37:
  %v240 = phi float [ %v508, %bb35 ], [ 0.0, %bb36 ]
  %v241 = call float @fp4_e2m1_to_f32(i8 %v152) #0
  br label %bb86
bb38:
  br label %bb40
bb39:
  br label %bb40
bb40:
  %v242 = phi float [ %v520, %bb38 ], [ 0.0, %bb39 ]
  %v243 = call float @fp4_e2m1_to_f32(i8 %v156) #0
  br label %bb88
bb41:
  br label %bb43
bb42:
  br label %bb43
bb43:
  %v244 = phi float [ %v532, %bb41 ], [ 0.0, %bb42 ]
  %v245 = fmul contract float %v230, %v165
  %v246 = fadd contract float %v65, %v245
  %v247 = fmul contract float %v232, %v174
  %v248 = fadd contract float %v66, %v247
  %v249 = fmul contract float %v234, %v183
  %v250 = fadd contract float %v246, %v249
  %v251 = fmul contract float %v236, %v192
  %v252 = fadd contract float %v248, %v251
  %v253 = fmul contract float %v238, %v201
  %v254 = fadd contract float %v250, %v253
  %v255 = fmul contract float %v240, %v210
  %v256 = fadd contract float %v252, %v255
  %v257 = fmul contract float %v242, %v219
  %v258 = fadd contract float %v254, %v257
  %v259 = fmul contract float %v244, %v228
  %v260 = fadd contract float %v256, %v259
  %v261 = add i64 %v93, 4
  %v262 = icmp ult i64 %v261, %v94
  %v263 = extractvalue { ptr, i64 } %v22, 0
  %v264 = getelementptr inbounds i8, ptr %v263, i64 %v261
  %v265 = load i8, ptr %v264, align 1
  %v266 = zext i8 %v265 to i32
  %v267 = add i64 %v261, 1
  %v268 = icmp ult i64 %v267, %v94
  %v269 = extractvalue { ptr, i64 } %v22, 0
  %v270 = getelementptr inbounds i8, ptr %v269, i64 %v267
  %v271 = load i8, ptr %v270, align 1
  %v272 = zext i8 %v271 to i32
  %v273 = add i64 %v261, 2
  %v274 = icmp ult i64 %v273, %v94
  %v275 = extractvalue { ptr, i64 } %v22, 0
  %v276 = getelementptr inbounds i8, ptr %v275, i64 %v273
  %v277 = load i8, ptr %v276, align 1
  %v278 = zext i8 %v277 to i32
  %v279 = add i64 %v261, 3
  %v280 = icmp ult i64 %v279, %v94
  %v281 = extractvalue { ptr, i64 } %v22, 0
  %v282 = getelementptr inbounds i8, ptr %v281, i64 %v279
  %v283 = load i8, ptr %v282, align 1
  %v284 = zext i8 %v283 to i32
  %v285 = and i32 8, 31
  %v286 = shl i32 %v272, %v285
  %v287 = or i32 %v266, %v286
  %v288 = and i32 16, 31
  %v289 = shl i32 %v278, %v288
  %v290 = or i32 %v287, %v289
  %v291 = and i32 24, 31
  %v292 = shl i32 %v284, %v291
  %v293 = or i32 %v290, %v292
  %v294 = and i32 %v293, 15
  %v295 = trunc i32 %v294 to i8
  %v296 = and i32 4, 31
  %v297 = lshr i32 %v293, %v296
  %v298 = and i32 %v297, 15
  %v299 = trunc i32 %v298 to i8
  %v300 = and i32 8, 31
  %v301 = lshr i32 %v293, %v300
  %v302 = and i32 %v301, 15
  %v303 = trunc i32 %v302 to i8
  %v304 = and i32 12, 31
  %v305 = lshr i32 %v293, %v304
  %v306 = and i32 %v305, 15
  %v307 = trunc i32 %v306 to i8
  %v308 = and i32 16, 31
  %v309 = lshr i32 %v293, %v308
  %v310 = and i32 %v309, 15
  %v311 = trunc i32 %v310 to i8
  %v312 = and i32 20, 31
  %v313 = lshr i32 %v293, %v312
  %v314 = and i32 %v313, 15
  %v315 = trunc i32 %v314 to i8
  %v316 = and i32 24, 31
  %v317 = lshr i32 %v293, %v316
  %v318 = and i32 %v317, 15
  %v319 = trunc i32 %v318 to i8
  %v320 = and i32 28, 31
  %v321 = lshr i32 %v293, %v320
  %v322 = and i32 %v321, 15
  %v323 = trunc i32 %v322 to i8
  %v324 = add i64 %v73, 8
  %v325 = icmp ult i64 %v324, %v157
  %v326 = extractvalue { ptr, i64 } %v24, 0
  %v327 = getelementptr inbounds i16, ptr %v326, i64 %v324
  %v328 = load i16, ptr %v327, align 2
  %v329 = zext i16 %v328 to i32
  %v330 = and i32 16, 31
  %v331 = shl i32 %v329, %v330
  %v332 = bitcast i32 %v331 to float
  %v333 = add i64 %v73, 9
  %v334 = icmp ult i64 %v333, %v157
  %v335 = extractvalue { ptr, i64 } %v24, 0
  %v336 = getelementptr inbounds i16, ptr %v335, i64 %v333
  %v337 = load i16, ptr %v336, align 2
  %v338 = zext i16 %v337 to i32
  %v339 = and i32 16, 31
  %v340 = shl i32 %v338, %v339
  %v341 = bitcast i32 %v340 to float
  %v342 = add i64 %v73, 10
  %v343 = icmp ult i64 %v342, %v157
  %v344 = extractvalue { ptr, i64 } %v24, 0
  %v345 = getelementptr inbounds i16, ptr %v344, i64 %v342
  %v346 = load i16, ptr %v345, align 2
  %v347 = zext i16 %v346 to i32
  %v348 = and i32 16, 31
  %v349 = shl i32 %v347, %v348
  %v350 = bitcast i32 %v349 to float
  %v351 = add i64 %v73, 11
  %v352 = icmp ult i64 %v351, %v157
  %v353 = extractvalue { ptr, i64 } %v24, 0
  %v354 = getelementptr inbounds i16, ptr %v353, i64 %v351
  %v355 = load i16, ptr %v354, align 2
  %v356 = zext i16 %v355 to i32
  %v357 = and i32 16, 31
  %v358 = shl i32 %v356, %v357
  %v359 = bitcast i32 %v358 to float
  %v360 = add i64 %v73, 12
  %v361 = icmp ult i64 %v360, %v157
  %v362 = extractvalue { ptr, i64 } %v24, 0
  %v363 = getelementptr inbounds i16, ptr %v362, i64 %v360
  %v364 = load i16, ptr %v363, align 2
  %v365 = zext i16 %v364 to i32
  %v366 = and i32 16, 31
  %v367 = shl i32 %v365, %v366
  %v368 = bitcast i32 %v367 to float
  %v369 = add i64 %v73, 13
  %v370 = icmp ult i64 %v369, %v157
  %v371 = extractvalue { ptr, i64 } %v24, 0
  %v372 = getelementptr inbounds i16, ptr %v371, i64 %v369
  %v373 = load i16, ptr %v372, align 2
  %v374 = zext i16 %v373 to i32
  %v375 = and i32 16, 31
  %v376 = shl i32 %v374, %v375
  %v377 = bitcast i32 %v376 to float
  %v378 = add i64 %v73, 14
  %v379 = icmp ult i64 %v378, %v157
  %v380 = extractvalue { ptr, i64 } %v24, 0
  %v381 = getelementptr inbounds i16, ptr %v380, i64 %v378
  %v382 = load i16, ptr %v381, align 2
  %v383 = zext i16 %v382 to i32
  %v384 = and i32 16, 31
  %v385 = shl i32 %v383, %v384
  %v386 = bitcast i32 %v385 to float
  %v387 = add i64 %v73, 15
  %v388 = icmp ult i64 %v387, %v157
  %v389 = extractvalue { ptr, i64 } %v24, 0
  %v390 = getelementptr inbounds i16, ptr %v389, i64 %v387
  %v391 = load i16, ptr %v390, align 2
  %v392 = zext i16 %v391 to i32
  %v393 = and i32 16, 31
  %v394 = shl i32 %v392, %v393
  %v395 = bitcast i32 %v394 to float
  %v396 = call float @fp4_e2m1_to_f32(i8 %v295) #0
  br label %bb90
bb44:
  br label %bb46
bb45:
  br label %bb46
bb46:
  %v397 = phi float [ %v544, %bb44 ], [ 0.0, %bb45 ]
  %v398 = call float @fp4_e2m1_to_f32(i8 %v299) #0
  br label %bb92
bb47:
  br label %bb49
bb48:
  br label %bb49
bb49:
  %v399 = phi float [ %v556, %bb47 ], [ 0.0, %bb48 ]
  %v400 = call float @fp4_e2m1_to_f32(i8 %v303) #0
  br label %bb94
bb50:
  br label %bb52
bb51:
  br label %bb52
bb52:
  %v401 = phi float [ %v568, %bb50 ], [ 0.0, %bb51 ]
  %v402 = call float @fp4_e2m1_to_f32(i8 %v307) #0
  br label %bb96
bb53:
  br label %bb55
bb54:
  br label %bb55
bb55:
  %v403 = phi float [ %v580, %bb53 ], [ 0.0, %bb54 ]
  %v404 = call float @fp4_e2m1_to_f32(i8 %v311) #0
  br label %bb98
bb56:
  br label %bb58
bb57:
  br label %bb58
bb58:
  %v405 = phi float [ %v592, %bb56 ], [ 0.0, %bb57 ]
  %v406 = call float @fp4_e2m1_to_f32(i8 %v315) #0
  br label %bb100
bb59:
  br label %bb61
bb60:
  br label %bb61
bb61:
  %v407 = phi float [ %v604, %bb59 ], [ 0.0, %bb60 ]
  %v408 = call float @fp4_e2m1_to_f32(i8 %v319) #0
  br label %bb102
bb62:
  br label %bb64
bb63:
  br label %bb64
bb64:
  %v409 = phi float [ %v616, %bb62 ], [ 0.0, %bb63 ]
  %v410 = call float @fp4_e2m1_to_f32(i8 %v323) #0
  br label %bb104
bb65:
  br label %bb67
bb66:
  br label %bb67
bb67:
  %v411 = phi float [ %v628, %bb65 ], [ 0.0, %bb66 ]
  %v412 = fmul contract float %v397, %v332
  %v413 = fadd contract float %v67, %v412
  %v414 = fmul contract float %v399, %v341
  %v415 = fadd contract float %v68, %v414
  %v416 = fmul contract float %v401, %v350
  %v417 = fadd contract float %v413, %v416
  %v418 = fmul contract float %v403, %v359
  %v419 = fadd contract float %v415, %v418
  %v420 = fmul contract float %v405, %v368
  %v421 = fadd contract float %v417, %v420
  %v422 = fmul contract float %v407, %v377
  %v423 = fadd contract float %v419, %v422
  %v424 = fmul contract float %v409, %v386
  %v425 = fadd contract float %v421, %v424
  %v426 = fmul contract float %v411, %v395
  %v427 = fadd contract float %v423, %v426
  br label %bb15
bb68:
  %v428 = extractvalue { ptr, i64 } %v21, 0
  %v429 = getelementptr inbounds float, ptr %v428, i64 %v86
  store float %v84, ptr %v429, align 4
  br label %bb69
bb69:
  ret void
bb70:
  %v430 = add i64 %v69, 1
  %v431 = insertvalue { i64, i64 } undef, i64 1, 0
  %v432 = insertvalue { i64, i64 } %v431, i64 %v69, 1
  br label %bb72
bb71:
  %v433 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb72
bb72:
  %v434 = phi { i64, i64 } [ %v432, %bb70 ], [ %v433, %bb71 ]
  %v435 = phi i64 [ %v430, %bb70 ], [ %v69, %bb71 ]
  %v436 = extractvalue { i64, i64 } %v434, 0
  %v437 = bitcast i64 %v436 to i64
  %v438 = icmp eq i64 %v437, 0
  br i1 %v438, label %bb18, label %bb73
bb73:
  %v439 = icmp eq i64 %v437, 1
  br i1 %v439, label %bb17, label %bb16
bb74:
  %v440 = fmul contract float %v229, %v89
  %v441 = bitcast float %v440 to i32
  %v442 = and i32 16, 31
  %v443 = lshr i32 %v441, %v442
  %v444 = trunc i32 %v443 to i16
  %v445 = zext i16 %v444 to i32
  %v446 = and i32 16, 31
  %v447 = shl i32 %v445, %v446
  %v448 = bitcast i32 %v447 to float
  %v449 = call float @__nv_fabsf(float %v448) #0
  br label %bb75
bb75:
  %v450 = fcmp olt float %v449, 0x7FF0000000000000
  %v451 = xor i1 %v450, 1
  br i1 %v451, label %bb21, label %bb20
bb76:
  %v452 = fmul contract float %v231, %v89
  %v453 = bitcast float %v452 to i32
  %v454 = and i32 16, 31
  %v455 = lshr i32 %v453, %v454
  %v456 = trunc i32 %v455 to i16
  %v457 = zext i16 %v456 to i32
  %v458 = and i32 16, 31
  %v459 = shl i32 %v457, %v458
  %v460 = bitcast i32 %v459 to float
  %v461 = call float @__nv_fabsf(float %v460) #0
  br label %bb77
bb77:
  %v462 = fcmp olt float %v461, 0x7FF0000000000000
  %v463 = xor i1 %v462, 1
  br i1 %v463, label %bb24, label %bb23
bb78:
  %v464 = fmul contract float %v233, %v89
  %v465 = bitcast float %v464 to i32
  %v466 = and i32 16, 31
  %v467 = lshr i32 %v465, %v466
  %v468 = trunc i32 %v467 to i16
  %v469 = zext i16 %v468 to i32
  %v470 = and i32 16, 31
  %v471 = shl i32 %v469, %v470
  %v472 = bitcast i32 %v471 to float
  %v473 = call float @__nv_fabsf(float %v472) #0
  br label %bb79
bb79:
  %v474 = fcmp olt float %v473, 0x7FF0000000000000
  %v475 = xor i1 %v474, 1
  br i1 %v475, label %bb27, label %bb26
bb80:
  %v476 = fmul contract float %v235, %v89
  %v477 = bitcast float %v476 to i32
  %v478 = and i32 16, 31
  %v479 = lshr i32 %v477, %v478
  %v480 = trunc i32 %v479 to i16
  %v481 = zext i16 %v480 to i32
  %v482 = and i32 16, 31
  %v483 = shl i32 %v481, %v482
  %v484 = bitcast i32 %v483 to float
  %v485 = call float @__nv_fabsf(float %v484) #0
  br label %bb81
bb81:
  %v486 = fcmp olt float %v485, 0x7FF0000000000000
  %v487 = xor i1 %v486, 1
  br i1 %v487, label %bb30, label %bb29
bb82:
  %v488 = fmul contract float %v237, %v89
  %v489 = bitcast float %v488 to i32
  %v490 = and i32 16, 31
  %v491 = lshr i32 %v489, %v490
  %v492 = trunc i32 %v491 to i16
  %v493 = zext i16 %v492 to i32
  %v494 = and i32 16, 31
  %v495 = shl i32 %v493, %v494
  %v496 = bitcast i32 %v495 to float
  %v497 = call float @__nv_fabsf(float %v496) #0
  br label %bb83
bb83:
  %v498 = fcmp olt float %v497, 0x7FF0000000000000
  %v499 = xor i1 %v498, 1
  br i1 %v499, label %bb33, label %bb32
bb84:
  %v500 = fmul contract float %v239, %v89
  %v501 = bitcast float %v500 to i32
  %v502 = and i32 16, 31
  %v503 = lshr i32 %v501, %v502
  %v504 = trunc i32 %v503 to i16
  %v505 = zext i16 %v504 to i32
  %v506 = and i32 16, 31
  %v507 = shl i32 %v505, %v506
  %v508 = bitcast i32 %v507 to float
  %v509 = call float @__nv_fabsf(float %v508) #0
  br label %bb85
bb85:
  %v510 = fcmp olt float %v509, 0x7FF0000000000000
  %v511 = xor i1 %v510, 1
  br i1 %v511, label %bb36, label %bb35
bb86:
  %v512 = fmul contract float %v241, %v89
  %v513 = bitcast float %v512 to i32
  %v514 = and i32 16, 31
  %v515 = lshr i32 %v513, %v514
  %v516 = trunc i32 %v515 to i16
  %v517 = zext i16 %v516 to i32
  %v518 = and i32 16, 31
  %v519 = shl i32 %v517, %v518
  %v520 = bitcast i32 %v519 to float
  %v521 = call float @__nv_fabsf(float %v520) #0
  br label %bb87
bb87:
  %v522 = fcmp olt float %v521, 0x7FF0000000000000
  %v523 = xor i1 %v522, 1
  br i1 %v523, label %bb39, label %bb38
bb88:
  %v524 = fmul contract float %v243, %v89
  %v525 = bitcast float %v524 to i32
  %v526 = and i32 16, 31
  %v527 = lshr i32 %v525, %v526
  %v528 = trunc i32 %v527 to i16
  %v529 = zext i16 %v528 to i32
  %v530 = and i32 16, 31
  %v531 = shl i32 %v529, %v530
  %v532 = bitcast i32 %v531 to float
  %v533 = call float @__nv_fabsf(float %v532) #0
  br label %bb89
bb89:
  %v534 = fcmp olt float %v533, 0x7FF0000000000000
  %v535 = xor i1 %v534, 1
  br i1 %v535, label %bb42, label %bb41
bb90:
  %v536 = fmul contract float %v396, %v89
  %v537 = bitcast float %v536 to i32
  %v538 = and i32 16, 31
  %v539 = lshr i32 %v537, %v538
  %v540 = trunc i32 %v539 to i16
  %v541 = zext i16 %v540 to i32
  %v542 = and i32 16, 31
  %v543 = shl i32 %v541, %v542
  %v544 = bitcast i32 %v543 to float
  %v545 = call float @__nv_fabsf(float %v544) #0
  br label %bb91
bb91:
  %v546 = fcmp olt float %v545, 0x7FF0000000000000
  %v547 = xor i1 %v546, 1
  br i1 %v547, label %bb45, label %bb44
bb92:
  %v548 = fmul contract float %v398, %v89
  %v549 = bitcast float %v548 to i32
  %v550 = and i32 16, 31
  %v551 = lshr i32 %v549, %v550
  %v552 = trunc i32 %v551 to i16
  %v553 = zext i16 %v552 to i32
  %v554 = and i32 16, 31
  %v555 = shl i32 %v553, %v554
  %v556 = bitcast i32 %v555 to float
  %v557 = call float @__nv_fabsf(float %v556) #0
  br label %bb93
bb93:
  %v558 = fcmp olt float %v557, 0x7FF0000000000000
  %v559 = xor i1 %v558, 1
  br i1 %v559, label %bb48, label %bb47
bb94:
  %v560 = fmul contract float %v400, %v89
  %v561 = bitcast float %v560 to i32
  %v562 = and i32 16, 31
  %v563 = lshr i32 %v561, %v562
  %v564 = trunc i32 %v563 to i16
  %v565 = zext i16 %v564 to i32
  %v566 = and i32 16, 31
  %v567 = shl i32 %v565, %v566
  %v568 = bitcast i32 %v567 to float
  %v569 = call float @__nv_fabsf(float %v568) #0
  br label %bb95
bb95:
  %v570 = fcmp olt float %v569, 0x7FF0000000000000
  %v571 = xor i1 %v570, 1
  br i1 %v571, label %bb51, label %bb50
bb96:
  %v572 = fmul contract float %v402, %v89
  %v573 = bitcast float %v572 to i32
  %v574 = and i32 16, 31
  %v575 = lshr i32 %v573, %v574
  %v576 = trunc i32 %v575 to i16
  %v577 = zext i16 %v576 to i32
  %v578 = and i32 16, 31
  %v579 = shl i32 %v577, %v578
  %v580 = bitcast i32 %v579 to float
  %v581 = call float @__nv_fabsf(float %v580) #0
  br label %bb97
bb97:
  %v582 = fcmp olt float %v581, 0x7FF0000000000000
  %v583 = xor i1 %v582, 1
  br i1 %v583, label %bb54, label %bb53
bb98:
  %v584 = fmul contract float %v404, %v89
  %v585 = bitcast float %v584 to i32
  %v586 = and i32 16, 31
  %v587 = lshr i32 %v585, %v586
  %v588 = trunc i32 %v587 to i16
  %v589 = zext i16 %v588 to i32
  %v590 = and i32 16, 31
  %v591 = shl i32 %v589, %v590
  %v592 = bitcast i32 %v591 to float
  %v593 = call float @__nv_fabsf(float %v592) #0
  br label %bb99
bb99:
  %v594 = fcmp olt float %v593, 0x7FF0000000000000
  %v595 = xor i1 %v594, 1
  br i1 %v595, label %bb57, label %bb56
bb100:
  %v596 = fmul contract float %v406, %v89
  %v597 = bitcast float %v596 to i32
  %v598 = and i32 16, 31
  %v599 = lshr i32 %v597, %v598
  %v600 = trunc i32 %v599 to i16
  %v601 = zext i16 %v600 to i32
  %v602 = and i32 16, 31
  %v603 = shl i32 %v601, %v602
  %v604 = bitcast i32 %v603 to float
  %v605 = call float @__nv_fabsf(float %v604) #0
  br label %bb101
bb101:
  %v606 = fcmp olt float %v605, 0x7FF0000000000000
  %v607 = xor i1 %v606, 1
  br i1 %v607, label %bb60, label %bb59
bb102:
  %v608 = fmul contract float %v408, %v89
  %v609 = bitcast float %v608 to i32
  %v610 = and i32 16, 31
  %v611 = lshr i32 %v609, %v610
  %v612 = trunc i32 %v611 to i16
  %v613 = zext i16 %v612 to i32
  %v614 = and i32 16, 31
  %v615 = shl i32 %v613, %v614
  %v616 = bitcast i32 %v615 to float
  %v617 = call float @__nv_fabsf(float %v616) #0
  br label %bb103
bb103:
  %v618 = fcmp olt float %v617, 0x7FF0000000000000
  %v619 = xor i1 %v618, 1
  br i1 %v619, label %bb63, label %bb62
bb104:
  %v620 = fmul contract float %v410, %v89
  %v621 = bitcast float %v620 to i32
  %v622 = and i32 16, 31
  %v623 = lshr i32 %v621, %v622
  %v624 = trunc i32 %v623 to i16
  %v625 = zext i16 %v624 to i32
  %v626 = and i32 16, 31
  %v627 = shl i32 %v625, %v626
  %v628 = bitcast i32 %v627 to float
  %v629 = call float @__nv_fabsf(float %v628) #0
  br label %bb105
bb105:
  %v630 = fcmp olt float %v629, 0x7FF0000000000000
  %v631 = xor i1 %v630, 1
  br i1 %v631, label %bb66, label %bb65
bb106:
  unreachable
bb107:
  unreachable
bb108:
  unreachable
bb109:
  unreachable
}

define void @nvfp4_dequant_to_bf16(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, float %v6, i32 %v7, i32 %v8, i32 %v9) #0 {
entry:
  %v10 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v11 = insertvalue { ptr, i64 } %v10, i64 %v1, 1
  %v12 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v13 = insertvalue { ptr, i64 } %v12, i64 %v3, 1
  %v14 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v5, 1
  br label %bb0
bb0:
  %v16 = phi { ptr, i64 } [ %v11, %entry ]
  %v17 = phi { ptr, i64 } [ %v13, %entry ]
  %v18 = phi { ptr, i64 } [ %v15, %entry ]
  %v19 = phi float [ %v6, %entry ]
  %v20 = phi i32 [ %v7, %entry ]
  %v21 = phi i32 [ %v8, %entry ]
  %v22 = phi i32 [ %v9, %entry ]
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v24 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v25 = mul i32 %v23, %v24
  %v26 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v27 = add i32 %v25, %v26
  %v28 = zext i32 %v27 to i64
  %v29 = zext i32 %v20 to i64
  %v30 = icmp uge i64 %v28, %v29
  %v31 = xor i1 %v30, 1
  br i1 %v31, label %bb5, label %bb4
bb4:
  br label %bb17
bb5:
  %v32 = icmp eq i32 %v22, 0
  %v33 = xor i1 %v32, 1
  br i1 %v33, label %bb6, label %bb28
bb6:
  %v34 = udiv i32 %v21, %v22
  %v35 = zext i32 %v34 to i64
  %v36 = zext i32 %v21 to i64
  %v37 = zext i32 %v22 to i64
  br label %bb7
bb7:
  %v38 = phi i64 [ 0, %bb6 ], [ %v76, %bb15 ]
  %v39 = icmp ult i64 %v38, %v35
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb19, label %bb18
bb8:
  unreachable
bb9:
  %v41 = extractvalue { i64, i64 } %v75, 1
  %v42 = mul i64 %v28, %v35
  %v43 = add i64 %v42, %v41
  %v44 = extractvalue { ptr, i64 } %v18, 1
  %v45 = icmp ult i64 %v43, %v44
  br i1 %v45, label %bb11, label %bb29
bb10:
  br label %bb17
bb11:
  %v46 = extractvalue { ptr, i64 } %v18, 0
  %v47 = getelementptr inbounds i8, ptr %v46, i64 %v43
  %v48 = load i8, ptr %v47, align 1
  %v49 = call float @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v48) #0
  br label %bb12
bb12:
  %v50 = udiv i64 %v37, 2
  br label %bb13
bb13:
  %v51 = phi i64 [ 0, %bb12 ], [ %v86, %bb27 ]
  %v52 = icmp ult i64 %v51, %v50
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb23, label %bb22
bb14:
  %v54 = extractvalue { i64, i64 } %v85, 1
  %v55 = udiv i64 %v36, 2
  %v56 = mul i64 %v28, %v55
  %v57 = mul i64 %v41, %v37
  %v58 = udiv i64 %v57, 2
  %v59 = add i64 %v56, %v58
  %v60 = add i64 %v59, %v54
  %v61 = extractvalue { ptr, i64 } %v17, 1
  %v62 = icmp ult i64 %v60, %v61
  br i1 %v62, label %bb16, label %bb30
bb15:
  br label %bb7
bb16:
  %v63 = extractvalue { ptr, i64 } %v17, 0
  %v64 = getelementptr inbounds i8, ptr %v63, i64 %v60
  %v65 = load i8, ptr %v64, align 1
  %v66 = trunc i32 4 to i8
  %v67 = and i8 %v66, 7
  %v68 = lshr i8 %v65, %v67
  %v69 = and i8 %v68, 15
  %v70 = call float @fp4_e2m1_to_f32(i8 %v69) #0
  br label %bb26
bb17:
  ret void
bb18:
  %v71 = add i64 %v38, 1
  %v72 = insertvalue { i64, i64 } undef, i64 1, 0
  %v73 = insertvalue { i64, i64 } %v72, i64 %v38, 1
  br label %bb20
bb19:
  %v74 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb20
bb20:
  %v75 = phi { i64, i64 } [ %v73, %bb18 ], [ %v74, %bb19 ]
  %v76 = phi i64 [ %v71, %bb18 ], [ %v38, %bb19 ]
  %v77 = extractvalue { i64, i64 } %v75, 0
  %v78 = bitcast i64 %v77 to i64
  %v79 = icmp eq i64 %v78, 0
  br i1 %v79, label %bb10, label %bb21
bb21:
  %v80 = icmp eq i64 %v78, 1
  br i1 %v80, label %bb9, label %bb8
bb22:
  %v81 = add i64 %v51, 1
  %v82 = insertvalue { i64, i64 } undef, i64 1, 0
  %v83 = insertvalue { i64, i64 } %v82, i64 %v51, 1
  br label %bb24
bb23:
  %v84 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb24
bb24:
  %v85 = phi { i64, i64 } [ %v83, %bb22 ], [ %v84, %bb23 ]
  %v86 = phi i64 [ %v81, %bb22 ], [ %v51, %bb23 ]
  %v87 = extractvalue { i64, i64 } %v85, 0
  %v88 = bitcast i64 %v87 to i64
  %v89 = icmp eq i64 %v88, 0
  br i1 %v89, label %bb15, label %bb25
bb25:
  %v90 = icmp eq i64 %v88, 1
  br i1 %v90, label %bb14, label %bb8
bb26:
  %v91 = fmul contract float %v70, %v49
  %v92 = fdiv contract float %v91, %v19
  %v93 = and i8 %v65, 15
  %v94 = call float @fp4_e2m1_to_f32(i8 %v93) #0
  br label %bb27
bb27:
  %v95 = fmul contract float %v94, %v49
  %v96 = fdiv contract float %v95, %v19
  %v97 = mul i64 %v28, %v36
  %v98 = add i64 %v97, %v57
  %v99 = mul i64 %v54, 2
  %v100 = add i64 %v98, %v99
  %v101 = bitcast float %v96 to i32
  %v102 = and i32 16, 31
  %v103 = lshr i32 %v101, %v102
  %v104 = trunc i32 %v103 to i16
  %v105 = extractvalue { ptr, i64 } %v16, 0
  %v106 = getelementptr inbounds i16, ptr %v105, i64 %v100
  store i16 %v104, ptr %v106, align 2
  %v107 = bitcast float %v92 to i32
  %v108 = and i32 16, 31
  %v109 = lshr i32 %v107, %v108
  %v110 = trunc i32 %v109 to i16
  %v111 = add i64 %v100, 1
  %v112 = getelementptr inbounds i16, ptr %v105, i64 %v111
  store i16 %v110, ptr %v112, align 2
  br label %bb13
bb28:
  unreachable
bb29:
  unreachable
bb30:
  unreachable
}

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.y()

define void @nvfp4_gemm_fused(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, float %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12) #0 {
entry:
  %v13 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v5, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v7, 1
  br label %bb0
bb0:
  %v21 = phi { ptr, i64 } [ %v14, %entry ]
  %v22 = phi { ptr, i64 } [ %v16, %entry ]
  %v23 = phi { ptr, i64 } [ %v18, %entry ]
  %v24 = phi { ptr, i64 } [ %v20, %entry ]
  %v25 = phi float [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi i32 [ %v10, %entry ]
  %v28 = phi i32 [ %v11, %entry ]
  %v29 = phi i32 [ %v12, %entry ]
  %v30 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb1
bb1:
  %v31 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb2
bb2:
  %v32 = mul i32 %v30, %v31
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb3
bb3:
  %v34 = add i32 %v32, %v33
  %v35 = bitcast i32 %v34 to i32
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb4
bb4:
  %v37 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb5
bb5:
  %v38 = mul i32 %v36, %v37
  %v39 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb6
bb6:
  %v40 = add i32 %v38, %v39
  %v41 = bitcast i32 %v40 to i32
  %v42 = bitcast i32 %v26 to i32
  %v43 = icmp sge i32 %v35, %v42
  %v44 = xor i1 %v43, 1
  br i1 %v44, label %bb8, label %bb7
bb7:
  br label %bb10
bb8:
  %v45 = bitcast i32 %v27 to i32
  %v46 = icmp sge i32 %v41, %v45
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb11, label %bb9
bb9:
  br label %bb10
bb10:
  br label %bb29
bb11:
  %v48 = zext i32 %v28 to i64
  %v49 = zext i32 %v27 to i64
  %v50 = zext i32 %v29 to i64
  %v51 = icmp eq i64 %v50, 0
  %v52 = xor i1 %v51, 1
  br i1 %v52, label %bb12, label %bb44
bb12:
  %v53 = udiv i64 %v48, %v50
  br label %bb13
bb13:
  %v54 = phi float [ 0.0, %bb12 ], [ %v80, %bb21 ]
  %v55 = phi i64 [ 0, %bb12 ], [ %v160, %bb21 ]
  %v56 = icmp ult i64 %v55, %v53
  %v57 = xor i1 %v56, 1
  br i1 %v57, label %bb31, label %bb30
bb14:
  unreachable
bb15:
  %v58 = extractvalue { i64, i64 } %v159, 1
  %v59 = sext i32 %v41 to i64
  %v60 = mul i64 %v59, %v53
  %v61 = add i64 %v60, %v58
  %v62 = extractvalue { ptr, i64 } %v23, 1
  %v63 = icmp ult i64 %v61, %v62
  br i1 %v63, label %bb17, label %bb45
bb16:
  %v64 = bitcast float %v54 to i32
  %v65 = and i32 16, 31
  %v66 = lshr i32 %v64, %v65
  %v67 = trunc i32 %v66 to i16
  %v68 = sext i32 %v35 to i64
  %v69 = mul i64 %v68, %v49
  %v70 = sext i32 %v41 to i64
  %v71 = add i64 %v69, %v70
  %v72 = extractvalue { ptr, i64 } %v21, 0
  %v73 = getelementptr inbounds i16, ptr %v72, i64 %v71
  store i16 %v67, ptr %v73, align 2
  br label %bb29
bb17:
  %v74 = extractvalue { ptr, i64 } %v23, 0
  %v75 = getelementptr inbounds i8, ptr %v74, i64 %v61
  %v76 = load i8, ptr %v75, align 1
  %v77 = call float @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v76) #0
  br label %bb18
bb18:
  %v78 = fdiv contract float %v77, %v25
  %v79 = udiv i64 %v50, 8
  br label %bb19
bb19:
  %v80 = phi float [ %v54, %bb18 ], [ %v125, %bb24 ]
  %v81 = phi i64 [ 0, %bb18 ], [ %v170, %bb24 ]
  %v82 = icmp ult i64 %v81, %v79
  %v83 = xor i1 %v82, 1
  br i1 %v83, label %bb35, label %bb34
bb20:
  %v84 = extractvalue { i64, i64 } %v169, 1
  %v85 = udiv i64 %v48, 2
  %v86 = mul i64 %v59, %v85
  %v87 = mul i64 %v58, %v50
  %v88 = udiv i64 %v87, 2
  %v89 = add i64 %v86, %v88
  %v90 = mul i64 %v84, 4
  %v91 = add i64 %v89, %v90
  %v92 = extractvalue { ptr, i64 } %v22, 1
  %v93 = icmp ult i64 %v91, %v92
  %v94 = extractvalue { ptr, i64 } %v22, 0
  %v95 = getelementptr inbounds i8, ptr %v94, i64 %v91
  %v96 = load i8, ptr %v95, align 1
  %v97 = zext i8 %v96 to i32
  %v98 = add i64 %v91, 1
  %v99 = icmp ult i64 %v98, %v92
  %v100 = extractvalue { ptr, i64 } %v22, 0
  %v101 = getelementptr inbounds i8, ptr %v100, i64 %v98
  %v102 = load i8, ptr %v101, align 1
  %v103 = zext i8 %v102 to i32
  %v104 = add i64 %v91, 2
  %v105 = icmp ult i64 %v104, %v92
  %v106 = extractvalue { ptr, i64 } %v22, 0
  %v107 = getelementptr inbounds i8, ptr %v106, i64 %v104
  %v108 = load i8, ptr %v107, align 1
  %v109 = zext i8 %v108 to i32
  %v110 = add i64 %v91, 3
  %v111 = icmp ult i64 %v110, %v92
  %v112 = extractvalue { ptr, i64 } %v22, 0
  %v113 = getelementptr inbounds i8, ptr %v112, i64 %v110
  %v114 = load i8, ptr %v113, align 1
  %v115 = zext i8 %v114 to i32
  %v116 = and i32 8, 31
  %v117 = shl i32 %v103, %v116
  %v118 = or i32 %v97, %v117
  %v119 = and i32 16, 31
  %v120 = shl i32 %v109, %v119
  %v121 = or i32 %v118, %v120
  %v122 = and i32 24, 31
  %v123 = shl i32 %v115, %v122
  %v124 = or i32 %v121, %v123
  br label %bb22
bb21:
  br label %bb13
bb22:
  %v125 = phi float [ %v80, %bb20 ], [ %v154, %bb28 ]
  %v126 = phi i64 [ 0, %bb20 ], [ %v180, %bb28 ]
  %v127 = icmp ult i64 %v126, 8
  %v128 = xor i1 %v127, 1
  br i1 %v128, label %bb39, label %bb38
bb23:
  %v129 = extractvalue { i64, i64 } %v179, 1
  %v130 = mul i64 %v129, 4
  %v131 = trunc i64 %v130 to i32
  %v132 = and i32 %v131, 31
  %v133 = lshr i32 %v124, %v132
  %v134 = and i32 %v133, 15
  %v135 = trunc i32 %v134 to i8
  %v136 = call float @fp4_e2m1_to_f32(i8 %v135) #0
  br label %bb42
bb24:
  br label %bb19
bb25:
  br label %bb27
bb26:
  br label %bb27
bb27:
  %v137 = phi float [ %v193, %bb25 ], [ 0.0, %bb26 ]
  %v138 = mul i64 %v84, 8
  %v139 = add i64 %v87, %v138
  %v140 = add i64 %v139, %v129
  %v141 = sext i32 %v35 to i64
  %v142 = mul i64 %v141, %v48
  %v143 = add i64 %v142, %v140
  %v144 = extractvalue { ptr, i64 } %v24, 1
  %v145 = icmp ult i64 %v143, %v144
  br i1 %v145, label %bb28, label %bb46
bb28:
  %v146 = extractvalue { ptr, i64 } %v24, 0
  %v147 = getelementptr inbounds i16, ptr %v146, i64 %v143
  %v148 = load i16, ptr %v147, align 2
  %v149 = zext i16 %v148 to i32
  %v150 = and i32 16, 31
  %v151 = shl i32 %v149, %v150
  %v152 = bitcast i32 %v151 to float
  %v153 = fmul contract float %v137, %v152
  %v154 = fadd contract float %v125, %v153
  br label %bb22
bb29:
  ret void
bb30:
  %v155 = add i64 %v55, 1
  %v156 = insertvalue { i64, i64 } undef, i64 1, 0
  %v157 = insertvalue { i64, i64 } %v156, i64 %v55, 1
  br label %bb32
bb31:
  %v158 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb32
bb32:
  %v159 = phi { i64, i64 } [ %v157, %bb30 ], [ %v158, %bb31 ]
  %v160 = phi i64 [ %v155, %bb30 ], [ %v55, %bb31 ]
  %v161 = extractvalue { i64, i64 } %v159, 0
  %v162 = bitcast i64 %v161 to i64
  %v163 = icmp eq i64 %v162, 0
  br i1 %v163, label %bb16, label %bb33
bb33:
  %v164 = icmp eq i64 %v162, 1
  br i1 %v164, label %bb15, label %bb14
bb34:
  %v165 = add i64 %v81, 1
  %v166 = insertvalue { i64, i64 } undef, i64 1, 0
  %v167 = insertvalue { i64, i64 } %v166, i64 %v81, 1
  br label %bb36
bb35:
  %v168 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb36
bb36:
  %v169 = phi { i64, i64 } [ %v167, %bb34 ], [ %v168, %bb35 ]
  %v170 = phi i64 [ %v165, %bb34 ], [ %v81, %bb35 ]
  %v171 = extractvalue { i64, i64 } %v169, 0
  %v172 = bitcast i64 %v171 to i64
  %v173 = icmp eq i64 %v172, 0
  br i1 %v173, label %bb21, label %bb37
bb37:
  %v174 = icmp eq i64 %v172, 1
  br i1 %v174, label %bb20, label %bb14
bb38:
  %v175 = add i64 %v126, 1
  %v176 = insertvalue { i64, i64 } undef, i64 1, 0
  %v177 = insertvalue { i64, i64 } %v176, i64 %v126, 1
  br label %bb40
bb39:
  %v178 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb40
bb40:
  %v179 = phi { i64, i64 } [ %v177, %bb38 ], [ %v178, %bb39 ]
  %v180 = phi i64 [ %v175, %bb38 ], [ %v126, %bb39 ]
  %v181 = extractvalue { i64, i64 } %v179, 0
  %v182 = bitcast i64 %v181 to i64
  %v183 = icmp eq i64 %v182, 0
  br i1 %v183, label %bb24, label %bb41
bb41:
  %v184 = icmp eq i64 %v182, 1
  br i1 %v184, label %bb23, label %bb14
bb42:
  %v185 = fmul contract float %v136, %v78
  %v186 = bitcast float %v185 to i32
  %v187 = and i32 16, 31
  %v188 = lshr i32 %v186, %v187
  %v189 = trunc i32 %v188 to i16
  %v190 = zext i16 %v189 to i32
  %v191 = and i32 16, 31
  %v192 = shl i32 %v190, %v191
  %v193 = bitcast i32 %v192 to float
  %v194 = call float @__nv_fabsf(float %v193) #0
  br label %bb43
bb43:
  %v195 = fcmp olt float %v194, 0x7FF0000000000000
  %v196 = xor i1 %v195, 1
  br i1 %v196, label %bb26, label %bb25
bb44:
  unreachable
bb45:
  unreachable
bb46:
  unreachable
}

define void @nvfp4_gemm_fused_ksplit(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, ptr %v6, i64 %v7, float %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12) #0 {
entry:
  %v13 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v1, 1
  %v15 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v16 = insertvalue { ptr, i64 } %v15, i64 %v3, 1
  %v17 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v18 = insertvalue { ptr, i64 } %v17, i64 %v5, 1
  %v19 = insertvalue { ptr, i64 } undef, ptr %v6, 0
  %v20 = insertvalue { ptr, i64 } %v19, i64 %v7, 1
  br label %bb0
bb0:
  %v21 = phi { ptr, i64 } [ %v14, %entry ]
  %v22 = phi { ptr, i64 } [ %v16, %entry ]
  %v23 = phi { ptr, i64 } [ %v18, %entry ]
  %v24 = phi { ptr, i64 } [ %v20, %entry ]
  %v25 = phi float [ %v8, %entry ]
  %v26 = phi i32 [ %v9, %entry ]
  %v27 = phi i32 [ %v10, %entry ]
  %v28 = phi i32 [ %v11, %entry ]
  %v29 = phi i32 [ %v12, %entry ]
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v31 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v32 = mul i32 %v31, 64
  %v33 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v34 = add i32 %v32, %v33
  %v35 = zext i32 %v34 to i64
  %v36 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb4
bb4:
  %v37 = zext i32 %v36 to i64
  %v38 = zext i32 %v26 to i64
  %v39 = zext i32 %v27 to i64
  %v40 = zext i32 %v28 to i64
  %v41 = icmp uge i64 %v35, %v38
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb6, label %bb5
bb5:
  br label %bb39
bb6:
  %v43 = zext i32 %v29 to i64
  %v44 = add i64 %v39, %v43
  %v45 = sub i64 %v44, 1
  %v46 = icmp eq i64 %v43, 0
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb7, label %bb54
bb7:
  %v48 = udiv i64 %v45, %v43
  %v49 = mul i64 %v37, %v48
  %v50 = add i64 %v49, %v48
  %v51 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v50, i64 %v39) #0
  br label %bb8
bb8:
  %v52 = icmp eq i64 %v40, 0
  %v53 = xor i1 %v52, 1
  br i1 %v53, label %bb9, label %bb55
bb9:
  %v54 = udiv i64 %v49, %v40
  %v55 = mul i64 %v54, %v40
  %v56 = add i64 %v51, %v40
  %v57 = sub i64 %v56, 1
  %v58 = udiv i64 %v57, %v40
  %v59 = mul i64 %v58, %v40
  %v60 = call i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v59, i64 %v39) #0
  br label %bb10
bb10:
  %v61 = udiv i64 %v39, %v40
  %v62 = udiv i64 %v55, %v40
  %v63 = udiv i64 %v60, %v40
  br label %bb11
bb11:
  %v64 = phi float [ 0.0, %bb10 ], [ %v82, %bb36 ]
  %v65 = phi i64 [ %v62, %bb10 ], [ %v170, %bb36 ]
  %v66 = icmp ult i64 %v65, %v63
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb41, label %bb40
bb12:
  unreachable
bb13:
  %v68 = extractvalue { i64, i64 } %v169, 1
  %v69 = mul i64 %v68, %v40
  %v70 = icmp uge i64 %v69, %v51
  %v71 = xor i1 %v70, 1
  br i1 %v71, label %bb15, label %bb14
bb14:
  br label %bb37
bb15:
  %v72 = mul i64 %v35, %v61
  %v73 = add i64 %v72, %v68
  %v74 = extractvalue { ptr, i64 } %v23, 1
  %v75 = icmp ult i64 %v73, %v74
  br i1 %v75, label %bb16, label %bb56
bb16:
  %v76 = extractvalue { ptr, i64 } %v23, 0
  %v77 = getelementptr inbounds i8, ptr %v76, i64 %v73
  %v78 = load i8, ptr %v77, align 1
  %v79 = call float @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v78) #0
  br label %bb17
bb17:
  %v80 = fdiv contract float %v79, %v25
  %v81 = udiv i64 %v40, 8
  br label %bb18
bb18:
  %v82 = phi float [ %v64, %bb17 ], [ %v128, %bb28 ]
  %v83 = phi i64 [ 0, %bb17 ], [ %v180, %bb28 ]
  %v84 = icmp ult i64 %v83, %v81
  %v85 = xor i1 %v84, 1
  br i1 %v85, label %bb45, label %bb44
bb19:
  %v86 = extractvalue { i64, i64 } %v179, 1
  %v87 = udiv i64 %v39, 2
  %v88 = mul i64 %v35, %v87
  %v89 = udiv i64 %v69, 2
  %v90 = add i64 %v88, %v89
  %v91 = mul i64 %v86, 4
  %v92 = add i64 %v90, %v91
  %v93 = add i64 %v92, 3
  %v94 = extractvalue { ptr, i64 } %v22, 1
  %v95 = icmp uge i64 %v93, %v94
  %v96 = xor i1 %v95, 1
  br i1 %v96, label %bb21, label %bb20
bb20:
  br label %bb36
bb21:
  %v97 = icmp ult i64 %v92, %v94
  br i1 %v97, label %bb22, label %bb57
bb22:
  %v98 = extractvalue { ptr, i64 } %v22, 0
  %v99 = getelementptr inbounds i8, ptr %v98, i64 %v92
  %v100 = load i8, ptr %v99, align 1
  %v101 = zext i8 %v100 to i32
  %v102 = add i64 %v92, 1
  %v103 = icmp ult i64 %v102, %v94
  br i1 %v103, label %bb23, label %bb58
bb23:
  %v104 = extractvalue { ptr, i64 } %v22, 0
  %v105 = getelementptr inbounds i8, ptr %v104, i64 %v102
  %v106 = load i8, ptr %v105, align 1
  %v107 = zext i8 %v106 to i32
  %v108 = add i64 %v92, 2
  %v109 = icmp ult i64 %v108, %v94
  br i1 %v109, label %bb24, label %bb59
bb24:
  %v110 = extractvalue { ptr, i64 } %v22, 0
  %v111 = getelementptr inbounds i8, ptr %v110, i64 %v108
  %v112 = load i8, ptr %v111, align 1
  %v113 = zext i8 %v112 to i32
  %v114 = icmp ult i64 %v93, %v94
  br i1 %v114, label %bb25, label %bb60
bb25:
  %v115 = extractvalue { ptr, i64 } %v22, 0
  %v116 = getelementptr inbounds i8, ptr %v115, i64 %v93
  %v117 = load i8, ptr %v116, align 1
  %v118 = zext i8 %v117 to i32
  %v119 = and i32 8, 31
  %v120 = shl i32 %v107, %v119
  %v121 = or i32 %v101, %v120
  %v122 = and i32 16, 31
  %v123 = shl i32 %v113, %v122
  %v124 = or i32 %v121, %v123
  %v125 = and i32 24, 31
  %v126 = shl i32 %v118, %v125
  %v127 = or i32 %v124, %v126
  br label %bb26
bb26:
  %v128 = phi float [ %v82, %bb25 ], [ %v158, %bb34 ], [ %v128, %bb35 ]
  %v129 = phi i64 [ 0, %bb25 ], [ %v190, %bb34 ], [ %v190, %bb35 ]
  %v130 = icmp ult i64 %v129, 8
  %v131 = xor i1 %v130, 1
  br i1 %v131, label %bb49, label %bb48
bb27:
  %v132 = extractvalue { i64, i64 } %v189, 1
  %v133 = mul i64 %v86, 8
  %v134 = add i64 %v69, %v133
  %v135 = add i64 %v134, %v132
  %v136 = icmp ult i64 %v135, %v49
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb29, label %bb35
bb28:
  br label %bb18
bb29:
  %v138 = icmp uge i64 %v135, %v51
  %v139 = xor i1 %v138, 1
  br i1 %v139, label %bb30, label %bb35
bb30:
  %v140 = mul i64 %v132, 4
  %v141 = trunc i64 %v140 to i32
  %v142 = and i32 %v141, 31
  %v143 = lshr i32 %v127, %v142
  %v144 = and i32 %v143, 15
  %v145 = trunc i32 %v144 to i8
  %v146 = call float @fp4_e2m1_to_f32(i8 %v145) #0
  br label %bb52
bb31:
  br label %bb33
bb32:
  br label %bb33
bb33:
  %v147 = phi float [ %v203, %bb31 ], [ 0.0, %bb32 ]
  %v148 = extractvalue { ptr, i64 } %v24, 1
  %v149 = icmp ult i64 %v135, %v148
  br i1 %v149, label %bb34, label %bb61
bb34:
  %v150 = extractvalue { ptr, i64 } %v24, 0
  %v151 = getelementptr inbounds i16, ptr %v150, i64 %v135
  %v152 = load i16, ptr %v151, align 2
  %v153 = zext i16 %v152 to i32
  %v154 = and i32 16, 31
  %v155 = shl i32 %v153, %v154
  %v156 = bitcast i32 %v155 to float
  %v157 = fmul contract float %v147, %v156
  %v158 = fadd contract float %v128, %v157
  br label %bb26
bb35:
  br label %bb26
bb36:
  br label %bb11
bb37:
  %v159 = mul i64 %v37, %v38
  %v160 = add i64 %v159, %v35
  %v161 = extractvalue { ptr, i64 } %v21, 1
  %v162 = icmp ult i64 %v160, %v161
  br i1 %v162, label %bb38, label %bb62
bb38:
  %v163 = extractvalue { ptr, i64 } %v21, 0
  %v164 = getelementptr inbounds float, ptr %v163, i64 %v160
  store float %v64, ptr %v164, align 4
  br label %bb39
bb39:
  ret void
bb40:
  %v165 = add i64 %v65, 1
  %v166 = insertvalue { i64, i64 } undef, i64 1, 0
  %v167 = insertvalue { i64, i64 } %v166, i64 %v65, 1
  br label %bb42
bb41:
  %v168 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb42
bb42:
  %v169 = phi { i64, i64 } [ %v167, %bb40 ], [ %v168, %bb41 ]
  %v170 = phi i64 [ %v165, %bb40 ], [ %v65, %bb41 ]
  %v171 = extractvalue { i64, i64 } %v169, 0
  %v172 = bitcast i64 %v171 to i64
  %v173 = icmp eq i64 %v172, 0
  br i1 %v173, label %bb37, label %bb43
bb43:
  %v174 = icmp eq i64 %v172, 1
  br i1 %v174, label %bb13, label %bb12
bb44:
  %v175 = add i64 %v83, 1
  %v176 = insertvalue { i64, i64 } undef, i64 1, 0
  %v177 = insertvalue { i64, i64 } %v176, i64 %v83, 1
  br label %bb46
bb45:
  %v178 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb46
bb46:
  %v179 = phi { i64, i64 } [ %v177, %bb44 ], [ %v178, %bb45 ]
  %v180 = phi i64 [ %v175, %bb44 ], [ %v83, %bb45 ]
  %v181 = extractvalue { i64, i64 } %v179, 0
  %v182 = bitcast i64 %v181 to i64
  %v183 = icmp eq i64 %v182, 0
  br i1 %v183, label %bb36, label %bb47
bb47:
  %v184 = icmp eq i64 %v182, 1
  br i1 %v184, label %bb19, label %bb12
bb48:
  %v185 = add i64 %v129, 1
  %v186 = insertvalue { i64, i64 } undef, i64 1, 0
  %v187 = insertvalue { i64, i64 } %v186, i64 %v129, 1
  br label %bb50
bb49:
  %v188 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb50
bb50:
  %v189 = phi { i64, i64 } [ %v187, %bb48 ], [ %v188, %bb49 ]
  %v190 = phi i64 [ %v185, %bb48 ], [ %v129, %bb49 ]
  %v191 = extractvalue { i64, i64 } %v189, 0
  %v192 = bitcast i64 %v191 to i64
  %v193 = icmp eq i64 %v192, 0
  br i1 %v193, label %bb28, label %bb51
bb51:
  %v194 = icmp eq i64 %v192, 1
  br i1 %v194, label %bb27, label %bb12
bb52:
  %v195 = fmul contract float %v146, %v80
  %v196 = bitcast float %v195 to i32
  %v197 = and i32 16, 31
  %v198 = lshr i32 %v196, %v197
  %v199 = trunc i32 %v198 to i16
  %v200 = zext i16 %v199 to i32
  %v201 = and i32 16, 31
  %v202 = shl i32 %v200, %v201
  %v203 = bitcast i32 %v202 to float
  %v204 = call float @__nv_fabsf(float %v203) #0
  br label %bb53
bb53:
  %v205 = fcmp olt float %v204, 0x7FF0000000000000
  %v206 = xor i1 %v205, 1
  br i1 %v206, label %bb32, label %bb31
bb54:
  unreachable
bb55:
  unreachable
bb56:
  unreachable
bb57:
  unreachable
bb58:
  unreachable
bb59:
  unreachable
bb60:
  unreachable
bb61:
  unreachable
bb62:
  unreachable
}

define void @bf16_gemm_tiled(ptr %v0, i64 %v1, ptr %v2, i64 %v3, ptr %v4, i64 %v5, i32 %v6, i32 %v7, i32 %v8) #0 {
entry:
  %v9 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v10 = insertvalue { ptr, i64 } %v9, i64 %v1, 1
  %v11 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v12 = insertvalue { ptr, i64 } %v11, i64 %v3, 1
  %v13 = insertvalue { ptr, i64 } undef, ptr %v4, 0
  %v14 = insertvalue { ptr, i64 } %v13, i64 %v5, 1
  br label %bb0
bb0:
  %v15 = phi { ptr, i64 } [ %v10, %entry ]
  %v16 = phi { ptr, i64 } [ %v12, %entry ]
  %v17 = phi { ptr, i64 } [ %v14, %entry ]
  %v18 = phi i32 [ %v6, %entry ]
  %v19 = phi i32 [ %v7, %entry ]
  %v20 = phi i32 [ %v8, %entry ]
  %v21 = alloca [16 x float], align 4
  call void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0
  br label %bb1
bb1:
  %v23 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb2
bb2:
  %v24 = zext i32 %v23 to i64
  %v25 = urem i64 %v24, 16
  %v26 = udiv i64 %v24, 16
  %v27 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb3
bb3:
  %v28 = mul i32 %v27, 64
  %v29 = zext i32 %v28 to i64
  %v30 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb4
bb4:
  %v31 = mul i32 %v30, 64
  %v32 = zext i32 %v31 to i64
  %v33 = zext i32 %v18 to i64
  %v34 = zext i32 %v19 to i64
  %v35 = zext i32 %v20 to i64
  %v36 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 0
  store float 0.0, ptr %v36, align 4
  %v37 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 1
  store float 0.0, ptr %v37, align 4
  %v38 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 2
  store float 0.0, ptr %v38, align 4
  %v39 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 3
  store float 0.0, ptr %v39, align 4
  %v40 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 4
  store float 0.0, ptr %v40, align 4
  %v41 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 5
  store float 0.0, ptr %v41, align 4
  %v42 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 6
  store float 0.0, ptr %v42, align 4
  %v43 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 7
  store float 0.0, ptr %v43, align 4
  %v44 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 8
  store float 0.0, ptr %v44, align 4
  %v45 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 9
  store float 0.0, ptr %v45, align 4
  %v46 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 10
  store float 0.0, ptr %v46, align 4
  %v47 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 11
  store float 0.0, ptr %v47, align 4
  %v48 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 12
  store float 0.0, ptr %v48, align 4
  %v49 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 13
  store float 0.0, ptr %v49, align 4
  %v50 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 14
  store float 0.0, ptr %v50, align 4
  %v51 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 15
  store float 0.0, ptr %v51, align 4
  br label %bb5
bb5:
  %v52 = phi i64 [ 0, %bb4 ], [ %v296, %bb40 ]
  %v53 = icmp ult i64 %v52, %v35
  %v54 = xor i1 %v53, 1
  br i1 %v54, label %bb53, label %bb52
bb6:
  unreachable
bb7:
  %v55 = extractvalue { i64, i64 } %v295, 1
  %v56 = mul i64 %v26, 4
  %v57 = add i64 %v29, %v56
  %v58 = add i64 %v57, 1
  %v59 = add i64 %v57, 2
  %v60 = add i64 %v57, 3
  %v61 = mul i64 %v25, 4
  %v62 = add i64 %v32, %v61
  %v63 = add i64 %v62, 1
  %v64 = add i64 %v62, 2
  %v65 = add i64 %v62, 3
  %v66 = icmp ult i64 %v57, %v33
  %v67 = xor i1 %v66, 1
  br i1 %v67, label %bb11, label %bb9
bb8:
  br label %bb41
bb9:
  %v68 = mul i64 %v57, %v35
  %v69 = add i64 %v68, %v55
  %v70 = extractvalue { ptr, i64 } %v16, 1
  %v71 = icmp ult i64 %v69, %v70
  br i1 %v71, label %bb10, label %bb64
bb10:
  %v72 = extractvalue { ptr, i64 } %v16, 0
  %v73 = getelementptr inbounds i16, ptr %v72, i64 %v69
  %v74 = load i16, ptr %v73, align 2
  %v75 = zext i16 %v74 to i32
  %v76 = and i32 16, 31
  %v77 = shl i32 %v75, %v76
  %v78 = bitcast i32 %v77 to float
  br label %bb12
bb11:
  br label %bb12
bb12:
  %v79 = phi float [ %v78, %bb10 ], [ 0.0, %bb11 ]
  %v80 = icmp ult i64 %v58, %v33
  %v81 = xor i1 %v80, 1
  br i1 %v81, label %bb15, label %bb13
bb13:
  %v82 = mul i64 %v58, %v35
  %v83 = add i64 %v82, %v55
  %v84 = extractvalue { ptr, i64 } %v16, 1
  %v85 = icmp ult i64 %v83, %v84
  br i1 %v85, label %bb14, label %bb65
bb14:
  %v86 = extractvalue { ptr, i64 } %v16, 0
  %v87 = getelementptr inbounds i16, ptr %v86, i64 %v83
  %v88 = load i16, ptr %v87, align 2
  %v89 = zext i16 %v88 to i32
  %v90 = and i32 16, 31
  %v91 = shl i32 %v89, %v90
  %v92 = bitcast i32 %v91 to float
  br label %bb16
bb15:
  br label %bb16
bb16:
  %v93 = phi float [ %v92, %bb14 ], [ 0.0, %bb15 ]
  %v94 = icmp ult i64 %v59, %v33
  %v95 = xor i1 %v94, 1
  br i1 %v95, label %bb19, label %bb17
bb17:
  %v96 = mul i64 %v59, %v35
  %v97 = add i64 %v96, %v55
  %v98 = extractvalue { ptr, i64 } %v16, 1
  %v99 = icmp ult i64 %v97, %v98
  br i1 %v99, label %bb18, label %bb66
bb18:
  %v100 = extractvalue { ptr, i64 } %v16, 0
  %v101 = getelementptr inbounds i16, ptr %v100, i64 %v97
  %v102 = load i16, ptr %v101, align 2
  %v103 = zext i16 %v102 to i32
  %v104 = and i32 16, 31
  %v105 = shl i32 %v103, %v104
  %v106 = bitcast i32 %v105 to float
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v107 = phi float [ %v106, %bb18 ], [ 0.0, %bb19 ]
  %v108 = icmp ult i64 %v60, %v33
  %v109 = xor i1 %v108, 1
  br i1 %v109, label %bb23, label %bb21
bb21:
  %v110 = mul i64 %v60, %v35
  %v111 = add i64 %v110, %v55
  %v112 = extractvalue { ptr, i64 } %v16, 1
  %v113 = icmp ult i64 %v111, %v112
  br i1 %v113, label %bb22, label %bb67
bb22:
  %v114 = extractvalue { ptr, i64 } %v16, 0
  %v115 = getelementptr inbounds i16, ptr %v114, i64 %v111
  %v116 = load i16, ptr %v115, align 2
  %v117 = zext i16 %v116 to i32
  %v118 = and i32 16, 31
  %v119 = shl i32 %v117, %v118
  %v120 = bitcast i32 %v119 to float
  br label %bb24
bb23:
  br label %bb24
bb24:
  %v121 = phi float [ %v120, %bb22 ], [ 0.0, %bb23 ]
  %v122 = icmp ult i64 %v62, %v34
  %v123 = xor i1 %v122, 1
  br i1 %v123, label %bb27, label %bb25
bb25:
  %v124 = mul i64 %v62, %v35
  %v125 = add i64 %v124, %v55
  %v126 = extractvalue { ptr, i64 } %v17, 1
  %v127 = icmp ult i64 %v125, %v126
  br i1 %v127, label %bb26, label %bb68
bb26:
  %v128 = extractvalue { ptr, i64 } %v17, 0
  %v129 = getelementptr inbounds i16, ptr %v128, i64 %v125
  %v130 = load i16, ptr %v129, align 2
  %v131 = zext i16 %v130 to i32
  %v132 = and i32 16, 31
  %v133 = shl i32 %v131, %v132
  %v134 = bitcast i32 %v133 to float
  br label %bb28
bb27:
  br label %bb28
bb28:
  %v135 = phi float [ %v134, %bb26 ], [ 0.0, %bb27 ]
  %v136 = icmp ult i64 %v63, %v34
  %v137 = xor i1 %v136, 1
  br i1 %v137, label %bb31, label %bb29
bb29:
  %v138 = mul i64 %v63, %v35
  %v139 = add i64 %v138, %v55
  %v140 = extractvalue { ptr, i64 } %v17, 1
  %v141 = icmp ult i64 %v139, %v140
  br i1 %v141, label %bb30, label %bb69
bb30:
  %v142 = extractvalue { ptr, i64 } %v17, 0
  %v143 = getelementptr inbounds i16, ptr %v142, i64 %v139
  %v144 = load i16, ptr %v143, align 2
  %v145 = zext i16 %v144 to i32
  %v146 = and i32 16, 31
  %v147 = shl i32 %v145, %v146
  %v148 = bitcast i32 %v147 to float
  br label %bb32
bb31:
  br label %bb32
bb32:
  %v149 = phi float [ %v148, %bb30 ], [ 0.0, %bb31 ]
  %v150 = icmp ult i64 %v64, %v34
  %v151 = xor i1 %v150, 1
  br i1 %v151, label %bb35, label %bb33
bb33:
  %v152 = mul i64 %v64, %v35
  %v153 = add i64 %v152, %v55
  %v154 = extractvalue { ptr, i64 } %v17, 1
  %v155 = icmp ult i64 %v153, %v154
  br i1 %v155, label %bb34, label %bb70
bb34:
  %v156 = extractvalue { ptr, i64 } %v17, 0
  %v157 = getelementptr inbounds i16, ptr %v156, i64 %v153
  %v158 = load i16, ptr %v157, align 2
  %v159 = zext i16 %v158 to i32
  %v160 = and i32 16, 31
  %v161 = shl i32 %v159, %v160
  %v162 = bitcast i32 %v161 to float
  br label %bb36
bb35:
  br label %bb36
bb36:
  %v163 = phi float [ %v162, %bb34 ], [ 0.0, %bb35 ]
  %v164 = icmp ult i64 %v65, %v34
  %v165 = xor i1 %v164, 1
  br i1 %v165, label %bb39, label %bb37
bb37:
  %v166 = mul i64 %v65, %v35
  %v167 = add i64 %v166, %v55
  %v168 = extractvalue { ptr, i64 } %v17, 1
  %v169 = icmp ult i64 %v167, %v168
  br i1 %v169, label %bb38, label %bb71
bb38:
  %v170 = extractvalue { ptr, i64 } %v17, 0
  %v171 = getelementptr inbounds i16, ptr %v170, i64 %v167
  %v172 = load i16, ptr %v171, align 2
  %v173 = zext i16 %v172 to i32
  %v174 = and i32 16, 31
  %v175 = shl i32 %v173, %v174
  %v176 = bitcast i32 %v175 to float
  br label %bb40
bb39:
  br label %bb40
bb40:
  %v177 = phi float [ %v176, %bb38 ], [ 0.0, %bb39 ]
  %v178 = fmul contract float %v79, %v135
  %v179 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 0
  %v180 = load float, ptr %v179, align 4
  %v181 = fadd contract float %v180, %v178
  %v182 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 0
  store float %v181, ptr %v182, align 4
  %v183 = fmul contract float %v79, %v149
  %v184 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 1
  %v185 = load float, ptr %v184, align 4
  %v186 = fadd contract float %v185, %v183
  %v187 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 1
  store float %v186, ptr %v187, align 4
  %v188 = fmul contract float %v79, %v163
  %v189 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 2
  %v190 = load float, ptr %v189, align 4
  %v191 = fadd contract float %v190, %v188
  %v192 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 2
  store float %v191, ptr %v192, align 4
  %v193 = fmul contract float %v79, %v177
  %v194 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 3
  %v195 = load float, ptr %v194, align 4
  %v196 = fadd contract float %v195, %v193
  %v197 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 3
  store float %v196, ptr %v197, align 4
  %v198 = fmul contract float %v93, %v135
  %v199 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 4
  %v200 = load float, ptr %v199, align 4
  %v201 = fadd contract float %v200, %v198
  %v202 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 4
  store float %v201, ptr %v202, align 4
  %v203 = fmul contract float %v93, %v149
  %v204 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 5
  %v205 = load float, ptr %v204, align 4
  %v206 = fadd contract float %v205, %v203
  %v207 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 5
  store float %v206, ptr %v207, align 4
  %v208 = fmul contract float %v93, %v163
  %v209 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 6
  %v210 = load float, ptr %v209, align 4
  %v211 = fadd contract float %v210, %v208
  %v212 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 6
  store float %v211, ptr %v212, align 4
  %v213 = fmul contract float %v93, %v177
  %v214 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 7
  %v215 = load float, ptr %v214, align 4
  %v216 = fadd contract float %v215, %v213
  %v217 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 7
  store float %v216, ptr %v217, align 4
  %v218 = fmul contract float %v107, %v135
  %v219 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 8
  %v220 = load float, ptr %v219, align 4
  %v221 = fadd contract float %v220, %v218
  %v222 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 8
  store float %v221, ptr %v222, align 4
  %v223 = fmul contract float %v107, %v149
  %v224 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 9
  %v225 = load float, ptr %v224, align 4
  %v226 = fadd contract float %v225, %v223
  %v227 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 9
  store float %v226, ptr %v227, align 4
  %v228 = fmul contract float %v107, %v163
  %v229 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 10
  %v230 = load float, ptr %v229, align 4
  %v231 = fadd contract float %v230, %v228
  %v232 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 10
  store float %v231, ptr %v232, align 4
  %v233 = fmul contract float %v107, %v177
  %v234 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 11
  %v235 = load float, ptr %v234, align 4
  %v236 = fadd contract float %v235, %v233
  %v237 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 11
  store float %v236, ptr %v237, align 4
  %v238 = fmul contract float %v121, %v135
  %v239 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 12
  %v240 = load float, ptr %v239, align 4
  %v241 = fadd contract float %v240, %v238
  %v242 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 12
  store float %v241, ptr %v242, align 4
  %v243 = fmul contract float %v121, %v149
  %v244 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 13
  %v245 = load float, ptr %v244, align 4
  %v246 = fadd contract float %v245, %v243
  %v247 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 13
  store float %v246, ptr %v247, align 4
  %v248 = fmul contract float %v121, %v163
  %v249 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 14
  %v250 = load float, ptr %v249, align 4
  %v251 = fadd contract float %v250, %v248
  %v252 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 14
  store float %v251, ptr %v252, align 4
  %v253 = fmul contract float %v121, %v177
  %v254 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 15
  %v255 = load float, ptr %v254, align 4
  %v256 = fadd contract float %v255, %v253
  %v257 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 15
  store float %v256, ptr %v257, align 4
  br label %bb5
bb41:
  %v258 = phi i64 [ 0, %bb8 ], [ %v306, %bb46 ]
  %v259 = icmp ult i64 %v258, 4
  %v260 = xor i1 %v259, 1
  br i1 %v260, label %bb57, label %bb56
bb42:
  %v261 = extractvalue { i64, i64 } %v305, 1
  br label %bb44
bb43:
  ret void
bb44:
  %v262 = phi i64 [ 0, %bb42 ], [ %v316, %bb51 ]
  %v263 = icmp ult i64 %v262, 4
  %v264 = xor i1 %v263, 1
  br i1 %v264, label %bb61, label %bb60
bb45:
  %v265 = extractvalue { i64, i64 } %v315, 1
  %v266 = mul i64 %v26, 4
  %v267 = add i64 %v29, %v266
  %v268 = add i64 %v267, %v261
  %v269 = mul i64 %v25, 4
  %v270 = add i64 %v32, %v269
  %v271 = add i64 %v270, %v265
  %v272 = icmp ult i64 %v268, %v33
  %v273 = xor i1 %v272, 1
  br i1 %v273, label %bb51, label %bb47
bb46:
  br label %bb41
bb47:
  %v274 = icmp ult i64 %v271, %v34
  %v275 = xor i1 %v274, 1
  br i1 %v275, label %bb51, label %bb48
bb48:
  %v276 = mul i64 %v268, %v34
  %v277 = add i64 %v276, %v271
  %v278 = mul i64 %v261, 4
  %v279 = add i64 %v278, %v265
  %v280 = icmp ult i64 %v279, 16
  br i1 %v280, label %bb49, label %bb72
bb49:
  %v281 = getelementptr inbounds [16 x float], ptr %v21, i32 0, i64 %v279
  %v282 = load float, ptr %v281, align 4
  %v283 = bitcast float %v282 to i32
  %v284 = and i32 16, 31
  %v285 = lshr i32 %v283, %v284
  %v286 = trunc i32 %v285 to i16
  %v287 = extractvalue { ptr, i64 } %v15, 1
  %v288 = icmp ult i64 %v277, %v287
  br i1 %v288, label %bb50, label %bb73
bb50:
  %v289 = extractvalue { ptr, i64 } %v15, 0
  %v290 = getelementptr inbounds i16, ptr %v289, i64 %v277
  store i16 %v286, ptr %v290, align 2
  br label %bb51
bb51:
  br label %bb44
bb52:
  %v291 = add i64 %v52, 1
  %v292 = insertvalue { i64, i64 } undef, i64 1, 0
  %v293 = insertvalue { i64, i64 } %v292, i64 %v52, 1
  br label %bb54
bb53:
  %v294 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb54
bb54:
  %v295 = phi { i64, i64 } [ %v293, %bb52 ], [ %v294, %bb53 ]
  %v296 = phi i64 [ %v291, %bb52 ], [ %v52, %bb53 ]
  %v297 = extractvalue { i64, i64 } %v295, 0
  %v298 = bitcast i64 %v297 to i64
  %v299 = icmp eq i64 %v298, 0
  br i1 %v299, label %bb8, label %bb55
bb55:
  %v300 = icmp eq i64 %v298, 1
  br i1 %v300, label %bb7, label %bb6
bb56:
  %v301 = add i64 %v258, 1
  %v302 = insertvalue { i64, i64 } undef, i64 1, 0
  %v303 = insertvalue { i64, i64 } %v302, i64 %v258, 1
  br label %bb58
bb57:
  %v304 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb58
bb58:
  %v305 = phi { i64, i64 } [ %v303, %bb56 ], [ %v304, %bb57 ]
  %v306 = phi i64 [ %v301, %bb56 ], [ %v258, %bb57 ]
  %v307 = extractvalue { i64, i64 } %v305, 0
  %v308 = bitcast i64 %v307 to i64
  %v309 = icmp eq i64 %v308, 0
  br i1 %v309, label %bb43, label %bb59
bb59:
  %v310 = icmp eq i64 %v308, 1
  br i1 %v310, label %bb42, label %bb6
bb60:
  %v311 = add i64 %v262, 1
  %v312 = insertvalue { i64, i64 } undef, i64 1, 0
  %v313 = insertvalue { i64, i64 } %v312, i64 %v262, 1
  br label %bb62
bb61:
  %v314 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb62
bb62:
  %v315 = phi { i64, i64 } [ %v313, %bb60 ], [ %v314, %bb61 ]
  %v316 = phi i64 [ %v311, %bb60 ], [ %v262, %bb61 ]
  %v317 = extractvalue { i64, i64 } %v315, 0
  %v318 = bitcast i64 %v317 to i64
  %v319 = icmp eq i64 %v318, 0
  br i1 %v319, label %bb46, label %bb63
bb63:
  %v320 = icmp eq i64 %v318, 1
  br i1 %v320, label %bb45, label %bb6
bb64:
  unreachable
bb65:
  unreachable
bb66:
  unreachable
bb67:
  unreachable
bb68:
  unreachable
bb69:
  unreachable
bb70:
  unreachable
bb71:
  unreachable
bb72:
  unreachable
bb73:
  unreachable
}

declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32)

define void @int4_gemm_innerNtB2_9AutoRoundEB4_(ptr %v0, ptr %v1, i64 %v2, ptr %v3, i64 %v4, ptr %v5, i64 %v6, ptr %v7, i64 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13) alwaysinline #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v1, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v2, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v4, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v5, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v6, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v8, 1
  br label %bb0
bb0:
  %v22 = phi ptr [ %v0, %entry ]
  %v23 = phi { ptr, i64 } [ %v15, %entry ]
  %v24 = phi { ptr, i64 } [ %v17, %entry ]
  %v25 = phi { ptr, i64 } [ %v19, %entry ]
  %v26 = phi { ptr, i64 } [ %v21, %entry ]
  %v27 = phi i32 [ %v9, %entry ]
  %v28 = phi i32 [ %v10, %entry ]
  %v29 = phi i32 [ %v11, %entry ]
  %v30 = phi i32 [ %v12, %entry ]
  %v31 = phi i32 [ %v13, %entry ]
  %v32 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v33 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v34 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb1
bb1:
  %v35 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb2
bb2:
  %v36 = mul i32 %v34, %v35
  %v37 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb3
bb3:
  %v38 = add i32 %v36, %v37
  %v39 = bitcast i32 %v38 to i32
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb4
bb4:
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb5
bb5:
  %v42 = mul i32 %v40, %v41
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb6
bb6:
  %v44 = add i32 %v42, %v43
  %v45 = bitcast i32 %v44 to i32
  %v46 = icmp sge i32 %v39, %v27
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb7, label %bb8
bb7:
  %v48 = icmp sge i32 %v45, %v28
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb9, label %bb8
bb8:
  br label %bb39
bb9:
  %v50 = sext i32 %v29 to i64
  %v51 = sext i32 %v28 to i64
  %v52 = sext i32 %v30 to i64
  %v53 = insertvalue { i32, i32 } undef, i32 0, 0
  %v54 = insertvalue { i32, i32 } %v53, i32 %v29, 1
  %v55 = extractvalue { i32, i32 } %v54, 0
  %v56 = extractvalue { i32, i32 } %v54, 1
  %v57 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangelEE3newCsgBauY1x2eDL_17infers_kernel_lib(i32 %v55, i32 %v56, i64 %v52) #0
  br label %bb40
bb10:
  %v58 = phi float [ %v140, %bb29 ], [ 0.0, %bb40 ]
  %v59 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 2
  %v60 = load i1, ptr %v59, align 1
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb42, label %bb41
bb11:
  unreachable
bb12:
  %v62 = extractvalue { i32, i32 } %v201, 1
  %v63 = icmp eq i32 %v30, 0
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb14, label %bb59
bb13:
  %v65 = bitcast float %v58 to i32
  %v66 = and i32 16, 31
  %v67 = lshr i32 %v65, %v66
  %v68 = trunc i32 %v67 to i16
  %v69 = sext i32 %v39 to i64
  %v70 = mul i64 %v69, %v51
  %v71 = sext i32 %v45 to i64
  %v72 = add i64 %v70, %v71
  %v73 = load { ptr, i64 }, ptr %v22, align 8
  %v74 = extractvalue { ptr, i64 } %v73, 0
  %v75 = getelementptr inbounds i16, ptr %v74, i64 %v72
  store i16 %v68, ptr %v75, align 2
  br label %bb39
bb14:
  %v76 = icmp eq i32 %v30, 4294967295
  %v77 = icmp eq i32 %v62, 2147483648
  %v78 = and i1 %v76, %v77
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb15, label %bb60
bb15:
  %v80 = sdiv i32 %v62, %v30
  %v81 = sext i32 %v80 to i64
  %v82 = icmp ne i32 %v31, 0
  %v83 = icmp eq i32 %v31, 0
  br i1 %v83, label %bb18, label %bb16
bb16:
  %v84 = mul i64 %v81, %v51
  %v85 = sext i32 %v45 to i64
  %v86 = add i64 %v84, %v85
  %v87 = extractvalue { ptr, i64 } %v24, 1
  %v88 = icmp ult i64 %v86, %v87
  br i1 %v88, label %bb17, label %bb61
bb17:
  %v89 = extractvalue { ptr, i64 } %v24, 0
  %v90 = getelementptr inbounds i16, ptr %v89, i64 %v86
  %v91 = load i16, ptr %v90, align 2
  br label %bb21
bb18:
  %v92 = icmp eq i64 %v52, 0
  %v93 = xor i1 %v92, 1
  br i1 %v93, label %bb19, label %bb62
bb19:
  %v94 = udiv i64 %v50, %v52
  %v95 = sext i32 %v45 to i64
  %v96 = mul i64 %v95, %v94
  %v97 = add i64 %v96, %v81
  %v98 = extractvalue { ptr, i64 } %v24, 1
  %v99 = icmp ult i64 %v97, %v98
  br i1 %v99, label %bb20, label %bb63
bb20:
  %v100 = extractvalue { ptr, i64 } %v24, 0
  %v101 = getelementptr inbounds i16, ptr %v100, i64 %v97
  %v102 = load i16, ptr %v101, align 2
  br label %bb21
bb21:
  %v103 = phi i16 [ %v91, %bb17 ], [ %v102, %bb20 ]
  %v104 = call float @f16_to_f32(i16 %v103) #0
  br label %bb46
bb22:
  %v105 = add i64 %v51, 7
  %v106 = udiv i64 %v105, 8
  %v107 = mul i64 %v81, %v106
  %v108 = sext i32 %v45 to i64
  %v109 = udiv i64 %v108, 8
  %v110 = add i64 %v107, %v109
  %v111 = srem i32 %v45, 8
  %v112 = sext i32 %v111 to i64
  %v113 = mul i64 %v112, 4
  br label %bb25
bb23:
  %v114 = icmp eq i64 %v52, 0
  %v115 = xor i1 %v114, 1
  br i1 %v115, label %bb24, label %bb64
bb24:
  %v116 = udiv i64 %v50, %v52
  %v117 = sext i32 %v45 to i64
  %v118 = mul i64 %v117, %v116
  %v119 = add i64 %v118, %v81
  %v120 = udiv i64 %v119, 8
  %v121 = urem i64 %v119, 8
  %v122 = mul i64 %v121, 4
  br label %bb25
bb25:
  %v123 = phi i64 [ %v110, %bb22 ], [ %v120, %bb24 ]
  %v124 = phi i64 [ %v113, %bb22 ], [ %v122, %bb24 ]
  %v125 = extractvalue { ptr, i64 } %v25, 1
  %v126 = icmp ult i64 %v123, %v125
  br i1 %v126, label %bb26, label %bb65
bb26:
  %v127 = extractvalue { ptr, i64 } %v25, 0
  %v128 = getelementptr inbounds i32, ptr %v127, i64 %v123
  %v129 = load i32, ptr %v128, align 4
  %v130 = trunc i64 %v124 to i32
  %v131 = and i32 %v130, 31
  %v132 = lshr i32 %v129, %v131
  %v133 = and i32 %v132, 15
  %v134 = trunc i32 %v133 to i8
  %v135 = insertvalue { i32, i32 } undef, i32 0, 0
  %v136 = insertvalue { i32, i32 } %v135, i32 %v30, 1
  %v137 = extractvalue { i32, i32 } %v136, 0
  %v138 = extractvalue { i32, i32 } %v136, 1
  %v139 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangelEE3newCsgBauY1x2eDL_17infers_kernel_lib(i32 %v137, i32 %v138, i64 8) #0
  br label %bb47
bb27:
  %v140 = phi float [ %v166, %bb36 ], [ %v58, %bb47 ]
  %v141 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 2
  %v142 = load i1, ptr %v141, align 1
  %v143 = xor i1 %v142, 1
  br i1 %v143, label %bb49, label %bb48
bb28:
  %v144 = extractvalue { i32, i32 } %v212, 1
  %v145 = xor i1 %v82, 1
  br i1 %v145, label %bb31, label %bb30
bb29:
  br label %bb10
bb30:
  %v146 = add i32 %v62, %v144
  %v147 = and i32 3, 31
  %v148 = ashr i32 %v146, %v147
  %v149 = sext i32 %v148 to i64
  %v150 = mul i64 %v149, %v51
  %v151 = sext i32 %v45 to i64
  %v152 = add i64 %v150, %v151
  br label %bb32
bb31:
  %v153 = sext i32 %v45 to i64
  %v154 = mul i64 %v153, %v50
  %v155 = sext i32 %v62 to i64
  %v156 = add i64 %v154, %v155
  %v157 = sext i32 %v144 to i64
  %v158 = add i64 %v156, %v157
  %v159 = udiv i64 %v158, 8
  br label %bb32
bb32:
  %v160 = phi i64 [ %v152, %bb30 ], [ %v159, %bb31 ]
  %v161 = extractvalue { ptr, i64 } %v23, 1
  %v162 = icmp ult i64 %v160, %v161
  br i1 %v162, label %bb33, label %bb66
bb33:
  %v163 = extractvalue { ptr, i64 } %v23, 0
  %v164 = getelementptr inbounds i32, ptr %v163, i64 %v160
  %v165 = load i32, ptr %v164, align 4
  br label %bb34
bb34:
  %v166 = phi float [ %v140, %bb33 ], [ %v195, %bb38 ]
  %v167 = phi i32 [ 0, %bb33 ], [ %v223, %bb38 ]
  %v168 = icmp slt i32 %v167, 8
  %v169 = xor i1 %v168, 1
  br i1 %v169, label %bb54, label %bb53
bb35:
  %v170 = extractvalue { i32, i32 } %v222, 1
  %v171 = mul i32 %v170, 4
  %v172 = and i32 %v171, 31
  %v173 = lshr i32 %v165, %v172
  %v174 = and i32 %v173, 15
  %v175 = trunc i32 %v174 to i8
  %v176 = call float @_infers_kernel_lib__shared__AutoRound_as_infers_kernel_lib__shared__Dequantize___dequant(i8 %v175, i8 %v134, float %v104) #0
  br label %bb37
bb36:
  br label %bb27
bb37:
  %v177 = sext i32 %v39 to i64
  %v178 = mul i64 %v177, %v50
  %v179 = sext i32 %v62 to i64
  %v180 = add i64 %v178, %v179
  %v181 = sext i32 %v144 to i64
  %v182 = add i64 %v180, %v181
  %v183 = sext i32 %v170 to i64
  %v184 = add i64 %v182, %v183
  %v185 = extractvalue { ptr, i64 } %v26, 1
  %v186 = icmp ult i64 %v184, %v185
  br i1 %v186, label %bb38, label %bb67
bb38:
  %v187 = extractvalue { ptr, i64 } %v26, 0
  %v188 = getelementptr inbounds i16, ptr %v187, i64 %v184
  %v189 = load i16, ptr %v188, align 2
  %v190 = zext i16 %v189 to i32
  %v191 = and i32 16, 31
  %v192 = shl i32 %v190, %v191
  %v193 = bitcast i32 %v192 to float
  %v194 = fmul contract float %v176, %v193
  %v195 = fadd contract float %v166, %v194
  br label %bb34
bb39:
  ret void
bb40:
  store { { i32, i32 }, i64, i1, [7 x i8] } %v57, ptr %v32, align 8
  br label %bb10
bb41:
  br label %bb43
bb42:
  %v196 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 1
  %v197 = load i64, ptr %v196, align 8
  br label %bb43
bb43:
  %v198 = phi i64 [ 0, %bb41 ], [ %v197, %bb42 ]
  %v199 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 2
  store i1 0, ptr %v199, align 1
  %v200 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 0
  %v201 = call { i32, i32 } @_RNvXs3_NtNtCsiQ4CSjCKWVc_4core4iter5rangeINtNtNtB9_3ops5range5RangelENtB5_17RangeIteratorImpl8spec_nthCsgBauY1x2eDL_17infers_kernel_lib(ptr %v200, i64 %v198) #0
  br label %bb44
bb44:
  %v202 = extractvalue { i32, i32 } %v201, 0
  %v203 = zext i32 %v202 to i64
  %v204 = icmp eq i64 %v203, 0
  br i1 %v204, label %bb13, label %bb45
bb45:
  %v205 = icmp eq i64 %v203, 1
  br i1 %v205, label %bb12, label %bb11
bb46:
  %v206 = xor i1 %v82, 1
  br i1 %v206, label %bb23, label %bb22
bb47:
  store { { i32, i32 }, i64, i1, [7 x i8] } %v139, ptr %v33, align 8
  br label %bb27
bb48:
  br label %bb50
bb49:
  %v207 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 1
  %v208 = load i64, ptr %v207, align 8
  br label %bb50
bb50:
  %v209 = phi i64 [ 0, %bb48 ], [ %v208, %bb49 ]
  %v210 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 2
  store i1 0, ptr %v210, align 1
  %v211 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 0
  %v212 = call { i32, i32 } @_RNvXs3_NtNtCsiQ4CSjCKWVc_4core4iter5rangeINtNtNtB9_3ops5range5RangelENtB5_17RangeIteratorImpl8spec_nthCsgBauY1x2eDL_17infers_kernel_lib(ptr %v211, i64 %v209) #0
  br label %bb51
bb51:
  %v213 = extractvalue { i32, i32 } %v212, 0
  %v214 = zext i32 %v213 to i64
  %v215 = icmp eq i64 %v214, 0
  br i1 %v215, label %bb29, label %bb52
bb52:
  %v216 = icmp eq i64 %v214, 1
  br i1 %v216, label %bb28, label %bb11
bb53:
  %v217 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %v167, i32 1) #0
  %v218 = extractvalue { i32, i1 } %v217, 0
  %v219 = extractvalue { i32, i1 } %v217, 1
  %v220 = xor i1 %v219, 1
  br i1 %v220, label %bb58, label %bb57
bb54:
  %v221 = insertvalue { i32, i32 } undef, i32 0, 0
  br label %bb55
bb55:
  %v222 = phi { i32, i32 } [ %v221, %bb54 ], [ %v229, %bb58 ]
  %v223 = phi i32 [ %v167, %bb54 ], [ %v218, %bb58 ]
  %v224 = extractvalue { i32, i32 } %v222, 0
  %v225 = zext i32 %v224 to i64
  %v226 = icmp eq i64 %v225, 0
  br i1 %v226, label %bb36, label %bb56
bb56:
  %v227 = icmp eq i64 %v225, 1
  br i1 %v227, label %bb35, label %bb11
bb57:
  br label %bb11
bb58:
  %v228 = insertvalue { i32, i32 } undef, i32 1, 0
  %v229 = insertvalue { i32, i32 } %v228, i32 %v167, 1
  br label %bb55
bb59:
  unreachable
bb60:
  unreachable
bb61:
  unreachable
bb62:
  unreachable
bb63:
  unreachable
bb64:
  unreachable
bb65:
  unreachable
bb66:
  unreachable
bb67:
  unreachable
}

define void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm40_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0 {
entry:
  br label %bb0
bb0:
  ret void
}

define float @f16_to_f32(i16 %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi i16 [ %v0, %entry ]
  %v2 = trunc i32 15 to i16
  %v3 = and i16 %v2, 15
  %v4 = lshr i16 %v1, %v3
  %v5 = zext i16 %v4 to i32
  %v6 = trunc i32 10 to i16
  %v7 = and i16 %v6, 15
  %v8 = lshr i16 %v1, %v7
  %v9 = and i16 %v8, 31
  %v10 = zext i16 %v9 to i32
  %v11 = and i16 %v1, 1023
  %v12 = icmp eq i32 %v10, 0
  br i1 %v12, label %bb1, label %bb5
bb1:
  %v13 = zext i16 %v11 to i32
  %v14 = and i32 13, 31
  %v15 = shl i32 %v13, %v14
  %v16 = icmp eq i16 %v11, 0
  br i1 %v16, label %bb3, label %bb2
bb2:
  br label %bb4
bb3:
  br label %bb4
bb4:
  %v17 = phi i32 [ 113, %bb2 ], [ 0, %bb3 ]
  %v18 = and i32 31, 31
  %v19 = shl i32 %v5, %v18
  %v20 = and i32 23, 31
  %v21 = shl i32 %v17, %v20
  %v22 = or i32 %v19, %v21
  %v23 = or i32 %v22, %v15
  %v24 = bitcast i32 %v23 to float
  br label %bb9
bb5:
  %v25 = icmp eq i32 %v10, 31
  br i1 %v25, label %bb6, label %bb7
bb6:
  %v26 = and i32 31, 31
  %v27 = shl i32 %v5, %v26
  %v28 = or i32 %v27, 2139095040
  %v29 = bitcast i32 %v28 to float
  br label %bb8
bb7:
  %v30 = add i32 %v10, 112
  %v31 = and i32 31, 31
  %v32 = shl i32 %v5, %v31
  %v33 = and i32 23, 31
  %v34 = shl i32 %v30, %v33
  %v35 = or i32 %v32, %v34
  %v36 = zext i16 %v11 to i32
  %v37 = and i32 13, 31
  %v38 = shl i32 %v36, %v37
  %v39 = or i32 %v35, %v38
  %v40 = bitcast i32 %v39 to float
  br label %bb8
bb8:
  %v41 = phi float [ %v29, %bb6 ], [ %v40, %bb7 ]
  br label %bb9
bb9:
  %v42 = phi float [ %v24, %bb4 ], [ %v41, %bb8 ]
  ret float %v42
}

define { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v0, i64 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i64, i64 } undef, i64 %v0, 0
  %v4 = insertvalue { i64, i64 } %v3, i64 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i64, i64 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb2, label %bb1
bb1:
  %v8 = extractvalue { i64, i64 } %v5, 0
  %v9 = extractvalue { i64, i64 } %v5, 1
  %v10 = call { i64, i64 } @_std__ops__Range_usize__as_std__iter__adapters__step_by__SpecRangeSetup_std__ops__Range_usize_____setup(i64 %v8, i64 %v9, i64 %v6) #0
  br label %bb3
bb2:
  unreachable
bb3:
  %v11 = sub i64 %v6, 1
  %v12 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } undef, { i64, i64 } %v10, 0
  %v13 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } %v12, i64 %v11, 1
  %v14 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } %v13, i1 1, 2
  ret { { i64, i64 }, i64, i1, [7 x i8] } %v14
bb4:
  unreachable
bb5:
  unreachable
bb6:
  unreachable
}

define void @_RINvNtCsNeIiTwFOhn_11cuda_device6thread22___launch_bounds_configKm100_Km0_ECsgBauY1x2eDL_17infers_kernel_lib() #0 {
entry:
  br label %bb0
bb0:
  ret void
}

define i64 @_RNvYjNtNtCsiQ4CSjCKWVc_4core3cmp3Ord3minCsgBauY1x2eDL_17infers_kernel_lib(i64 %v0, i64 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i64 [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = alloca i64, align 8
  %v5 = alloca i64, align 8
  store i64 %v2, ptr %v4, align 8
  store i64 %v3, ptr %v5, align 8
  %v6 = bitcast ptr %v5 to ptr
  %v7 = bitcast ptr %v4 to ptr
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_usize___lt(ptr %v6, ptr %v7) #0
  br label %bb1
bb1:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb3, label %bb2
bb2:
  %v10 = load i64, ptr %v5, align 8
  br label %bb4
bb3:
  %v11 = load i64, ptr %v4, align 8
  br label %bb4
bb4:
  %v12 = phi i64 [ %v10, %bb2 ], [ %v11, %bb3 ]
  ret i64 %v12
bb5:
  unreachable
bb6:
  unreachable
bb7:
  unreachable
bb8:
  unreachable
}

define void @int4_gemm_innerNtB2_4GgufEB4_(ptr %v0, ptr %v1, i64 %v2, ptr %v3, i64 %v4, ptr %v5, i64 %v6, ptr %v7, i64 %v8, i32 %v9, i32 %v10, i32 %v11, i32 %v12, i32 %v13) alwaysinline #0 {
entry:
  %v14 = insertvalue { ptr, i64 } undef, ptr %v1, 0
  %v15 = insertvalue { ptr, i64 } %v14, i64 %v2, 1
  %v16 = insertvalue { ptr, i64 } undef, ptr %v3, 0
  %v17 = insertvalue { ptr, i64 } %v16, i64 %v4, 1
  %v18 = insertvalue { ptr, i64 } undef, ptr %v5, 0
  %v19 = insertvalue { ptr, i64 } %v18, i64 %v6, 1
  %v20 = insertvalue { ptr, i64 } undef, ptr %v7, 0
  %v21 = insertvalue { ptr, i64 } %v20, i64 %v8, 1
  br label %bb0
bb0:
  %v22 = phi ptr [ %v0, %entry ]
  %v23 = phi { ptr, i64 } [ %v15, %entry ]
  %v24 = phi { ptr, i64 } [ %v17, %entry ]
  %v25 = phi { ptr, i64 } [ %v19, %entry ]
  %v26 = phi { ptr, i64 } [ %v21, %entry ]
  %v27 = phi i32 [ %v9, %entry ]
  %v28 = phi i32 [ %v10, %entry ]
  %v29 = phi i32 [ %v11, %entry ]
  %v30 = phi i32 [ %v12, %entry ]
  %v31 = phi i32 [ %v13, %entry ]
  %v32 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v33 = alloca { { i32, i32 }, i64, i1, [7 x i8] }, align 8
  %v34 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.y() #0
  br label %bb1
bb1:
  %v35 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.y() #0
  br label %bb2
bb2:
  %v36 = mul i32 %v34, %v35
  %v37 = call i32 @llvm.nvvm.read.ptx.sreg.tid.y() #0
  br label %bb3
bb3:
  %v38 = add i32 %v36, %v37
  %v39 = bitcast i32 %v38 to i32
  %v40 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb4
bb4:
  %v41 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb5
bb5:
  %v42 = mul i32 %v40, %v41
  %v43 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb6
bb6:
  %v44 = add i32 %v42, %v43
  %v45 = bitcast i32 %v44 to i32
  %v46 = icmp sge i32 %v39, %v27
  %v47 = xor i1 %v46, 1
  br i1 %v47, label %bb7, label %bb8
bb7:
  %v48 = icmp sge i32 %v45, %v28
  %v49 = xor i1 %v48, 1
  br i1 %v49, label %bb9, label %bb8
bb8:
  br label %bb39
bb9:
  %v50 = sext i32 %v29 to i64
  %v51 = sext i32 %v28 to i64
  %v52 = sext i32 %v30 to i64
  %v53 = insertvalue { i32, i32 } undef, i32 0, 0
  %v54 = insertvalue { i32, i32 } %v53, i32 %v29, 1
  %v55 = extractvalue { i32, i32 } %v54, 0
  %v56 = extractvalue { i32, i32 } %v54, 1
  %v57 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangelEE3newCsgBauY1x2eDL_17infers_kernel_lib(i32 %v55, i32 %v56, i64 %v52) #0
  br label %bb40
bb10:
  %v58 = phi float [ %v140, %bb29 ], [ 0.0, %bb40 ]
  %v59 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 2
  %v60 = load i1, ptr %v59, align 1
  %v61 = xor i1 %v60, 1
  br i1 %v61, label %bb42, label %bb41
bb11:
  unreachable
bb12:
  %v62 = extractvalue { i32, i32 } %v201, 1
  %v63 = icmp eq i32 %v30, 0
  %v64 = xor i1 %v63, 1
  br i1 %v64, label %bb14, label %bb59
bb13:
  %v65 = bitcast float %v58 to i32
  %v66 = and i32 16, 31
  %v67 = lshr i32 %v65, %v66
  %v68 = trunc i32 %v67 to i16
  %v69 = sext i32 %v39 to i64
  %v70 = mul i64 %v69, %v51
  %v71 = sext i32 %v45 to i64
  %v72 = add i64 %v70, %v71
  %v73 = load { ptr, i64 }, ptr %v22, align 8
  %v74 = extractvalue { ptr, i64 } %v73, 0
  %v75 = getelementptr inbounds i16, ptr %v74, i64 %v72
  store i16 %v68, ptr %v75, align 2
  br label %bb39
bb14:
  %v76 = icmp eq i32 %v30, 4294967295
  %v77 = icmp eq i32 %v62, 2147483648
  %v78 = and i1 %v76, %v77
  %v79 = xor i1 %v78, 1
  br i1 %v79, label %bb15, label %bb60
bb15:
  %v80 = sdiv i32 %v62, %v30
  %v81 = sext i32 %v80 to i64
  %v82 = icmp ne i32 %v31, 0
  %v83 = icmp eq i32 %v31, 0
  br i1 %v83, label %bb18, label %bb16
bb16:
  %v84 = mul i64 %v81, %v51
  %v85 = sext i32 %v45 to i64
  %v86 = add i64 %v84, %v85
  %v87 = extractvalue { ptr, i64 } %v24, 1
  %v88 = icmp ult i64 %v86, %v87
  br i1 %v88, label %bb17, label %bb61
bb17:
  %v89 = extractvalue { ptr, i64 } %v24, 0
  %v90 = getelementptr inbounds i16, ptr %v89, i64 %v86
  %v91 = load i16, ptr %v90, align 2
  br label %bb21
bb18:
  %v92 = icmp eq i64 %v52, 0
  %v93 = xor i1 %v92, 1
  br i1 %v93, label %bb19, label %bb62
bb19:
  %v94 = udiv i64 %v50, %v52
  %v95 = sext i32 %v45 to i64
  %v96 = mul i64 %v95, %v94
  %v97 = add i64 %v96, %v81
  %v98 = extractvalue { ptr, i64 } %v24, 1
  %v99 = icmp ult i64 %v97, %v98
  br i1 %v99, label %bb20, label %bb63
bb20:
  %v100 = extractvalue { ptr, i64 } %v24, 0
  %v101 = getelementptr inbounds i16, ptr %v100, i64 %v97
  %v102 = load i16, ptr %v101, align 2
  br label %bb21
bb21:
  %v103 = phi i16 [ %v91, %bb17 ], [ %v102, %bb20 ]
  %v104 = call float @f16_to_f32(i16 %v103) #0
  br label %bb46
bb22:
  %v105 = add i64 %v51, 7
  %v106 = udiv i64 %v105, 8
  %v107 = mul i64 %v81, %v106
  %v108 = sext i32 %v45 to i64
  %v109 = udiv i64 %v108, 8
  %v110 = add i64 %v107, %v109
  %v111 = srem i32 %v45, 8
  %v112 = sext i32 %v111 to i64
  %v113 = mul i64 %v112, 4
  br label %bb25
bb23:
  %v114 = icmp eq i64 %v52, 0
  %v115 = xor i1 %v114, 1
  br i1 %v115, label %bb24, label %bb64
bb24:
  %v116 = udiv i64 %v50, %v52
  %v117 = sext i32 %v45 to i64
  %v118 = mul i64 %v117, %v116
  %v119 = add i64 %v118, %v81
  %v120 = udiv i64 %v119, 8
  %v121 = urem i64 %v119, 8
  %v122 = mul i64 %v121, 4
  br label %bb25
bb25:
  %v123 = phi i64 [ %v110, %bb22 ], [ %v120, %bb24 ]
  %v124 = phi i64 [ %v113, %bb22 ], [ %v122, %bb24 ]
  %v125 = extractvalue { ptr, i64 } %v25, 1
  %v126 = icmp ult i64 %v123, %v125
  br i1 %v126, label %bb26, label %bb65
bb26:
  %v127 = extractvalue { ptr, i64 } %v25, 0
  %v128 = getelementptr inbounds i32, ptr %v127, i64 %v123
  %v129 = load i32, ptr %v128, align 4
  %v130 = trunc i64 %v124 to i32
  %v131 = and i32 %v130, 31
  %v132 = lshr i32 %v129, %v131
  %v133 = and i32 %v132, 15
  %v134 = trunc i32 %v133 to i8
  %v135 = insertvalue { i32, i32 } undef, i32 0, 0
  %v136 = insertvalue { i32, i32 } %v135, i32 %v30, 1
  %v137 = extractvalue { i32, i32 } %v136, 0
  %v138 = extractvalue { i32, i32 } %v136, 1
  %v139 = call { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangelEE3newCsgBauY1x2eDL_17infers_kernel_lib(i32 %v137, i32 %v138, i64 8) #0
  br label %bb47
bb27:
  %v140 = phi float [ %v166, %bb36 ], [ %v58, %bb47 ]
  %v141 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 2
  %v142 = load i1, ptr %v141, align 1
  %v143 = xor i1 %v142, 1
  br i1 %v143, label %bb49, label %bb48
bb28:
  %v144 = extractvalue { i32, i32 } %v212, 1
  %v145 = xor i1 %v82, 1
  br i1 %v145, label %bb31, label %bb30
bb29:
  br label %bb10
bb30:
  %v146 = add i32 %v62, %v144
  %v147 = and i32 3, 31
  %v148 = ashr i32 %v146, %v147
  %v149 = sext i32 %v148 to i64
  %v150 = mul i64 %v149, %v51
  %v151 = sext i32 %v45 to i64
  %v152 = add i64 %v150, %v151
  br label %bb32
bb31:
  %v153 = sext i32 %v45 to i64
  %v154 = mul i64 %v153, %v50
  %v155 = sext i32 %v62 to i64
  %v156 = add i64 %v154, %v155
  %v157 = sext i32 %v144 to i64
  %v158 = add i64 %v156, %v157
  %v159 = udiv i64 %v158, 8
  br label %bb32
bb32:
  %v160 = phi i64 [ %v152, %bb30 ], [ %v159, %bb31 ]
  %v161 = extractvalue { ptr, i64 } %v23, 1
  %v162 = icmp ult i64 %v160, %v161
  br i1 %v162, label %bb33, label %bb66
bb33:
  %v163 = extractvalue { ptr, i64 } %v23, 0
  %v164 = getelementptr inbounds i32, ptr %v163, i64 %v160
  %v165 = load i32, ptr %v164, align 4
  br label %bb34
bb34:
  %v166 = phi float [ %v140, %bb33 ], [ %v195, %bb38 ]
  %v167 = phi i32 [ 0, %bb33 ], [ %v223, %bb38 ]
  %v168 = icmp slt i32 %v167, 8
  %v169 = xor i1 %v168, 1
  br i1 %v169, label %bb54, label %bb53
bb35:
  %v170 = extractvalue { i32, i32 } %v222, 1
  %v171 = mul i32 %v170, 4
  %v172 = and i32 %v171, 31
  %v173 = lshr i32 %v165, %v172
  %v174 = and i32 %v173, 15
  %v175 = trunc i32 %v174 to i8
  %v176 = call float @_infers_kernel_lib__shared__Gguf_as_infers_kernel_lib__shared__Dequantize___dequant(i8 %v175, i8 %v134, float %v104) #0
  br label %bb37
bb36:
  br label %bb27
bb37:
  %v177 = sext i32 %v39 to i64
  %v178 = mul i64 %v177, %v50
  %v179 = sext i32 %v62 to i64
  %v180 = add i64 %v178, %v179
  %v181 = sext i32 %v144 to i64
  %v182 = add i64 %v180, %v181
  %v183 = sext i32 %v170 to i64
  %v184 = add i64 %v182, %v183
  %v185 = extractvalue { ptr, i64 } %v26, 1
  %v186 = icmp ult i64 %v184, %v185
  br i1 %v186, label %bb38, label %bb67
bb38:
  %v187 = extractvalue { ptr, i64 } %v26, 0
  %v188 = getelementptr inbounds i16, ptr %v187, i64 %v184
  %v189 = load i16, ptr %v188, align 2
  %v190 = zext i16 %v189 to i32
  %v191 = and i32 16, 31
  %v192 = shl i32 %v190, %v191
  %v193 = bitcast i32 %v192 to float
  %v194 = fmul contract float %v176, %v193
  %v195 = fadd contract float %v166, %v194
  br label %bb34
bb39:
  ret void
bb40:
  store { { i32, i32 }, i64, i1, [7 x i8] } %v57, ptr %v32, align 8
  br label %bb10
bb41:
  br label %bb43
bb42:
  %v196 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 1
  %v197 = load i64, ptr %v196, align 8
  br label %bb43
bb43:
  %v198 = phi i64 [ 0, %bb41 ], [ %v197, %bb42 ]
  %v199 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 2
  store i1 0, ptr %v199, align 1
  %v200 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v32, i32 0, i32 0
  %v201 = call { i32, i32 } @_RNvXs3_NtNtCsiQ4CSjCKWVc_4core4iter5rangeINtNtNtB9_3ops5range5RangelENtB5_17RangeIteratorImpl8spec_nthCsgBauY1x2eDL_17infers_kernel_lib(ptr %v200, i64 %v198) #0
  br label %bb44
bb44:
  %v202 = extractvalue { i32, i32 } %v201, 0
  %v203 = zext i32 %v202 to i64
  %v204 = icmp eq i64 %v203, 0
  br i1 %v204, label %bb13, label %bb45
bb45:
  %v205 = icmp eq i64 %v203, 1
  br i1 %v205, label %bb12, label %bb11
bb46:
  %v206 = xor i1 %v82, 1
  br i1 %v206, label %bb23, label %bb22
bb47:
  store { { i32, i32 }, i64, i1, [7 x i8] } %v139, ptr %v33, align 8
  br label %bb27
bb48:
  br label %bb50
bb49:
  %v207 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 1
  %v208 = load i64, ptr %v207, align 8
  br label %bb50
bb50:
  %v209 = phi i64 [ 0, %bb48 ], [ %v208, %bb49 ]
  %v210 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 2
  store i1 0, ptr %v210, align 1
  %v211 = getelementptr inbounds { { i32, i32 }, i64, i1, [7 x i8] }, ptr %v33, i32 0, i32 0
  %v212 = call { i32, i32 } @_RNvXs3_NtNtCsiQ4CSjCKWVc_4core4iter5rangeINtNtNtB9_3ops5range5RangelENtB5_17RangeIteratorImpl8spec_nthCsgBauY1x2eDL_17infers_kernel_lib(ptr %v211, i64 %v209) #0
  br label %bb51
bb51:
  %v213 = extractvalue { i32, i32 } %v212, 0
  %v214 = zext i32 %v213 to i64
  %v215 = icmp eq i64 %v214, 0
  br i1 %v215, label %bb29, label %bb52
bb52:
  %v216 = icmp eq i64 %v214, 1
  br i1 %v216, label %bb28, label %bb11
bb53:
  %v217 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %v167, i32 1) #0
  %v218 = extractvalue { i32, i1 } %v217, 0
  %v219 = extractvalue { i32, i1 } %v217, 1
  %v220 = xor i1 %v219, 1
  br i1 %v220, label %bb58, label %bb57
bb54:
  %v221 = insertvalue { i32, i32 } undef, i32 0, 0
  br label %bb55
bb55:
  %v222 = phi { i32, i32 } [ %v221, %bb54 ], [ %v229, %bb58 ]
  %v223 = phi i32 [ %v167, %bb54 ], [ %v218, %bb58 ]
  %v224 = extractvalue { i32, i32 } %v222, 0
  %v225 = zext i32 %v224 to i64
  %v226 = icmp eq i64 %v225, 0
  br i1 %v226, label %bb36, label %bb56
bb56:
  %v227 = icmp eq i64 %v225, 1
  br i1 %v227, label %bb35, label %bb11
bb57:
  br label %bb11
bb58:
  %v228 = insertvalue { i32, i32 } undef, i32 1, 0
  %v229 = insertvalue { i32, i32 } %v228, i32 %v167, 1
  br label %bb55
bb59:
  unreachable
bb60:
  unreachable
bb61:
  unreachable
bb62:
  unreachable
bb63:
  unreachable
bb64:
  unreachable
bb65:
  unreachable
bb66:
  unreachable
bb67:
  unreachable
}

define { i64, i64 } @_RNvXsc_NtNtCsiQ4CSjCKWVc_4core4iter5rangeINtNtNtB9_3ops5range14RangeInclusivejENtB5_26RangeInclusiveIteratorImpl9spec_nextCsgBauY1x2eDL_17infers_kernel_lib(ptr %v0) #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 2
  %v3 = load i1, ptr %v2, align 1
  %v4 = xor i1 %v3, 1
  br i1 %v4, label %bb10, label %bb9
bb1:
  %v5 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb8
bb2:
  %v6 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 0
  %v7 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 1
  %v8 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_usize___lt(ptr %v6, ptr %v7) #0
  br label %bb3
bb3:
  %v9 = xor i1 %v8, 1
  br i1 %v9, label %bb6, label %bb4
bb4:
  %v10 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 0
  %v11 = load i64, ptr %v10, align 8
  %v12 = call i64 @_usize_as_std__iter__Step___forward_unchecked(i64 %v11, i64 1) #0
  br label %bb5
bb5:
  %v13 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 0
  %v14 = load i64, ptr %v13, align 8
  %v15 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 0
  store i64 %v12, ptr %v15, align 8
  br label %bb7
bb6:
  %v16 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 2
  store i1 1, ptr %v16, align 1
  %v17 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 0
  %v18 = load i64, ptr %v17, align 8
  br label %bb7
bb7:
  %v19 = phi i64 [ %v14, %bb5 ], [ %v18, %bb6 ]
  %v20 = insertvalue { i64, i64 } undef, i64 1, 0
  %v21 = insertvalue { i64, i64 } %v20, i64 %v19, 1
  br label %bb8
bb8:
  %v22 = phi { i64, i64 } [ %v5, %bb1 ], [ %v21, %bb7 ]
  ret { i64, i64 } %v22
bb9:
  br label %bb1
bb10:
  %v23 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 0
  %v24 = getelementptr inbounds { i64, i64, i1, [7 x i8] }, ptr %v1, i32 0, i32 1
  %v25 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_usize___le(ptr %v23, ptr %v24) #0
  br label %bb11
bb11:
  %v26 = xor i1 %v25, 1
  %v27 = xor i1 %v26, 1
  br i1 %v27, label %bb2, label %bb1
}

define i64 @cuda_device____internal__index_1d(ptr %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi ptr [ %v0, %entry ]
  %v2 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb1
bb1:
  %v3 = zext i32 %v2 to i64
  %v4 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb2
bb2:
  %v5 = zext i32 %v4 to i64
  %v6 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb3
bb3:
  %v7 = zext i32 %v6 to i64
  %v8 = mul i64 %v5, %v7
  %v9 = add i64 %v8, %v3
  ret i64 %v9
}

define float @dev_sqrtf(float %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi float [ %v0, %entry ]
  %v2 = call float @__nv_sqrtf(float %v1) #0
  br label %bb1
bb1:
  ret float %v2
}

define void @fp8_dequantize_innerNtB2_7Fp8E4M3EB4_(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) alwaysinline #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v15 = mul i32 %v13, %v14
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v17 = add i32 %v15, %v16
  %v18 = zext i32 %v17 to i64
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb4
bb4:
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb5
bb5:
  %v21 = mul i32 %v19, %v20
  %v22 = zext i32 %v21 to i64
  %v23 = zext i32 %v11 to i64
  %v24 = insertvalue { i64, i64 } undef, i64 %v18, 0
  %v25 = insertvalue { i64, i64 } %v24, i64 %v23, 1
  %v26 = extractvalue { i64, i64 } %v25, 0
  %v27 = extractvalue { i64, i64 } %v25, 1
  %v28 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v26, i64 %v27, i64 %v22) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v28, ptr %v12, align 8
  br label %bb12
bb6:
  %v29 = phi i64 [ %v73, %bb11 ], [ %v59, %bb12 ]
  %v30 = phi i64 [ %v74, %bb11 ], [ %v62, %bb12 ]
  %v31 = add i64 %v64, 1
  %v32 = icmp eq i64 %v31, 0
  %v33 = select i1 %v32, i8 0, i8 1
  %v34 = insertvalue { i8, { { i64 } } } undef, i8 %v33, 0
  %v35 = insertvalue { i8, { { i64 } } } %v34, i64 %v31, 1, 0, 0
  %v36 = extractvalue { i8, { { i64 } } } %v35, 0
  %v37 = zext i8 %v36 to i64
  %v38 = icmp eq i64 %v37, 1
  %v39 = extractvalue { i8, { { i64 } } } %v35, 1
  %v40 = alloca { { i64 } }, align 8
  store { { i64 } } %v39, ptr %v40, align 8
  %v41 = load i64, ptr %v40, align 8
  %v42 = icmp ugt i64 %v30, 0
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb14, label %bb13
bb7:
  unreachable
bb8:
  %v44 = extractvalue { i64, i64 } %v72, 1
  %v45 = extractvalue { ptr, i64 } %v9, 1
  %v46 = icmp ult i64 %v44, %v45
  br i1 %v46, label %bb10, label %bb17
bb9:
  ret void
bb10:
  %v47 = extractvalue { ptr, i64 } %v9, 0
  %v48 = getelementptr inbounds i8, ptr %v47, i64 %v44
  %v49 = load i8, ptr %v48, align 1
  %v50 = call float @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v49) #0
  br label %bb11
bb11:
  %v51 = bitcast float %v50 to i32
  %v52 = and i32 16, 31
  %v53 = lshr i32 %v51, %v52
  %v54 = trunc i32 %v53 to i16
  %v55 = extractvalue { ptr, i64 } %v10, 0
  %v56 = getelementptr inbounds i16, ptr %v55, i64 %v44
  store i16 %v54, ptr %v56, align 2
  br label %bb6
bb12:
  %v57 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v58 = getelementptr inbounds { i64, i64 }, ptr %v57, i32 0, i32 0
  %v59 = load i64, ptr %v58, align 8
  %v60 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v61 = getelementptr inbounds { i64, i64 }, ptr %v60, i32 0, i32 1
  %v62 = load i64, ptr %v61, align 8
  %v63 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 1
  %v64 = load i64, ptr %v63, align 8
  %v65 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 2
  %v66 = load i1, ptr %v65, align 1
  br label %bb6
bb13:
  %v67 = add i64 %v29, %v41
  %v68 = sub i64 %v30, 1
  %v69 = insertvalue { i64, i64 } undef, i64 1, 0
  %v70 = insertvalue { i64, i64 } %v69, i64 %v29, 1
  br label %bb15
bb14:
  %v71 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb15
bb15:
  %v72 = phi { i64, i64 } [ %v70, %bb13 ], [ %v71, %bb14 ]
  %v73 = phi i64 [ %v67, %bb13 ], [ %v29, %bb14 ]
  %v74 = phi i64 [ %v68, %bb13 ], [ %v30, %bb14 ]
  %v75 = extractvalue { i64, i64 } %v72, 0
  %v76 = bitcast i64 %v75 to i64
  %v77 = icmp eq i64 %v76, 0
  br i1 %v77, label %bb9, label %bb16
bb16:
  %v78 = icmp eq i64 %v76, 1
  br i1 %v78, label %bb8, label %bb7
bb17:
  unreachable
}

define void @fp8_quantize_innerNtB2_7Fp8E5M2EB4_(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) alwaysinline #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v15 = mul i32 %v13, %v14
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v17 = add i32 %v15, %v16
  %v18 = zext i32 %v17 to i64
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb4
bb4:
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb5
bb5:
  %v21 = mul i32 %v19, %v20
  %v22 = zext i32 %v21 to i64
  %v23 = zext i32 %v11 to i64
  %v24 = insertvalue { i64, i64 } undef, i64 %v18, 0
  %v25 = insertvalue { i64, i64 } %v24, i64 %v23, 1
  %v26 = extractvalue { i64, i64 } %v25, 0
  %v27 = extractvalue { i64, i64 } %v25, 1
  %v28 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v26, i64 %v27, i64 %v22) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v28, ptr %v12, align 8
  br label %bb12
bb6:
  %v29 = phi i64 [ %v73, %bb11 ], [ %v59, %bb12 ]
  %v30 = phi i64 [ %v74, %bb11 ], [ %v62, %bb12 ]
  %v31 = add i64 %v64, 1
  %v32 = icmp eq i64 %v31, 0
  %v33 = select i1 %v32, i8 0, i8 1
  %v34 = insertvalue { i8, { { i64 } } } undef, i8 %v33, 0
  %v35 = insertvalue { i8, { { i64 } } } %v34, i64 %v31, 1, 0, 0
  %v36 = extractvalue { i8, { { i64 } } } %v35, 0
  %v37 = zext i8 %v36 to i64
  %v38 = icmp eq i64 %v37, 1
  %v39 = extractvalue { i8, { { i64 } } } %v35, 1
  %v40 = alloca { { i64 } }, align 8
  store { { i64 } } %v39, ptr %v40, align 8
  %v41 = load i64, ptr %v40, align 8
  %v42 = icmp ugt i64 %v30, 0
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb14, label %bb13
bb7:
  unreachable
bb8:
  %v44 = extractvalue { i64, i64 } %v72, 1
  %v45 = extractvalue { ptr, i64 } %v9, 1
  %v46 = icmp ult i64 %v44, %v45
  br i1 %v46, label %bb10, label %bb17
bb9:
  ret void
bb10:
  %v47 = extractvalue { ptr, i64 } %v9, 0
  %v48 = getelementptr inbounds i16, ptr %v47, i64 %v44
  %v49 = load i16, ptr %v48, align 2
  %v50 = zext i16 %v49 to i32
  %v51 = and i32 16, 31
  %v52 = shl i32 %v50, %v51
  %v53 = bitcast i32 %v52 to float
  %v54 = call i8 @_infers_kernel_lib__shared__Fp8E5M2_as_infers_kernel_lib__shared__Fp8Format___quantize(float %v53) #0
  br label %bb11
bb11:
  %v55 = extractvalue { ptr, i64 } %v10, 0
  %v56 = getelementptr inbounds i8, ptr %v55, i64 %v44
  store i8 %v54, ptr %v56, align 1
  br label %bb6
bb12:
  %v57 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v58 = getelementptr inbounds { i64, i64 }, ptr %v57, i32 0, i32 0
  %v59 = load i64, ptr %v58, align 8
  %v60 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v61 = getelementptr inbounds { i64, i64 }, ptr %v60, i32 0, i32 1
  %v62 = load i64, ptr %v61, align 8
  %v63 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 1
  %v64 = load i64, ptr %v63, align 8
  %v65 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 2
  %v66 = load i1, ptr %v65, align 1
  br label %bb6
bb13:
  %v67 = add i64 %v29, %v41
  %v68 = sub i64 %v30, 1
  %v69 = insertvalue { i64, i64 } undef, i64 1, 0
  %v70 = insertvalue { i64, i64 } %v69, i64 %v29, 1
  br label %bb15
bb14:
  %v71 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb15
bb15:
  %v72 = phi { i64, i64 } [ %v70, %bb13 ], [ %v71, %bb14 ]
  %v73 = phi i64 [ %v67, %bb13 ], [ %v29, %bb14 ]
  %v74 = phi i64 [ %v68, %bb13 ], [ %v30, %bb14 ]
  %v75 = extractvalue { i64, i64 } %v72, 0
  %v76 = bitcast i64 %v75 to i64
  %v77 = icmp eq i64 %v76, 0
  br i1 %v77, label %bb9, label %bb16
bb16:
  %v78 = icmp eq i64 %v76, 1
  br i1 %v78, label %bb8, label %bb7
bb17:
  unreachable
}

define void @fp8_quantize_innerNtB2_7Fp8E4M3EB4_(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) alwaysinline #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v15 = mul i32 %v13, %v14
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v17 = add i32 %v15, %v16
  %v18 = zext i32 %v17 to i64
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb4
bb4:
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb5
bb5:
  %v21 = mul i32 %v19, %v20
  %v22 = zext i32 %v21 to i64
  %v23 = zext i32 %v11 to i64
  %v24 = insertvalue { i64, i64 } undef, i64 %v18, 0
  %v25 = insertvalue { i64, i64 } %v24, i64 %v23, 1
  %v26 = extractvalue { i64, i64 } %v25, 0
  %v27 = extractvalue { i64, i64 } %v25, 1
  %v28 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v26, i64 %v27, i64 %v22) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v28, ptr %v12, align 8
  br label %bb12
bb6:
  %v29 = phi i64 [ %v73, %bb11 ], [ %v59, %bb12 ]
  %v30 = phi i64 [ %v74, %bb11 ], [ %v62, %bb12 ]
  %v31 = add i64 %v64, 1
  %v32 = icmp eq i64 %v31, 0
  %v33 = select i1 %v32, i8 0, i8 1
  %v34 = insertvalue { i8, { { i64 } } } undef, i8 %v33, 0
  %v35 = insertvalue { i8, { { i64 } } } %v34, i64 %v31, 1, 0, 0
  %v36 = extractvalue { i8, { { i64 } } } %v35, 0
  %v37 = zext i8 %v36 to i64
  %v38 = icmp eq i64 %v37, 1
  %v39 = extractvalue { i8, { { i64 } } } %v35, 1
  %v40 = alloca { { i64 } }, align 8
  store { { i64 } } %v39, ptr %v40, align 8
  %v41 = load i64, ptr %v40, align 8
  %v42 = icmp ugt i64 %v30, 0
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb14, label %bb13
bb7:
  unreachable
bb8:
  %v44 = extractvalue { i64, i64 } %v72, 1
  %v45 = extractvalue { ptr, i64 } %v9, 1
  %v46 = icmp ult i64 %v44, %v45
  br i1 %v46, label %bb10, label %bb17
bb9:
  ret void
bb10:
  %v47 = extractvalue { ptr, i64 } %v9, 0
  %v48 = getelementptr inbounds i16, ptr %v47, i64 %v44
  %v49 = load i16, ptr %v48, align 2
  %v50 = zext i16 %v49 to i32
  %v51 = and i32 16, 31
  %v52 = shl i32 %v50, %v51
  %v53 = bitcast i32 %v52 to float
  %v54 = call i8 @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___quantize(float %v53) #0
  br label %bb11
bb11:
  %v55 = extractvalue { ptr, i64 } %v10, 0
  %v56 = getelementptr inbounds i8, ptr %v55, i64 %v44
  store i8 %v54, ptr %v56, align 1
  br label %bb6
bb12:
  %v57 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v58 = getelementptr inbounds { i64, i64 }, ptr %v57, i32 0, i32 0
  %v59 = load i64, ptr %v58, align 8
  %v60 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v61 = getelementptr inbounds { i64, i64 }, ptr %v60, i32 0, i32 1
  %v62 = load i64, ptr %v61, align 8
  %v63 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 1
  %v64 = load i64, ptr %v63, align 8
  %v65 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 2
  %v66 = load i1, ptr %v65, align 1
  br label %bb6
bb13:
  %v67 = add i64 %v29, %v41
  %v68 = sub i64 %v30, 1
  %v69 = insertvalue { i64, i64 } undef, i64 1, 0
  %v70 = insertvalue { i64, i64 } %v69, i64 %v29, 1
  br label %bb15
bb14:
  %v71 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb15
bb15:
  %v72 = phi { i64, i64 } [ %v70, %bb13 ], [ %v71, %bb14 ]
  %v73 = phi i64 [ %v67, %bb13 ], [ %v29, %bb14 ]
  %v74 = phi i64 [ %v68, %bb13 ], [ %v30, %bb14 ]
  %v75 = extractvalue { i64, i64 } %v72, 0
  %v76 = bitcast i64 %v75 to i64
  %v77 = icmp eq i64 %v76, 0
  br i1 %v77, label %bb9, label %bb16
bb16:
  %v78 = icmp eq i64 %v76, 1
  br i1 %v78, label %bb8, label %bb7
bb17:
  unreachable
}

define void @fp8_dequantize_innerNtB2_7Fp8E5M2EB4_(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) alwaysinline #0 {
entry:
  %v5 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v6 = insertvalue { ptr, i64 } %v5, i64 %v1, 1
  %v7 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v8 = insertvalue { ptr, i64 } %v7, i64 %v3, 1
  br label %bb0
bb0:
  %v9 = phi { ptr, i64 } [ %v6, %entry ]
  %v10 = phi { ptr, i64 } [ %v8, %entry ]
  %v11 = phi i32 [ %v4, %entry ]
  %v12 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v13 = call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0
  br label %bb1
bb1:
  %v14 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v15 = mul i32 %v13, %v14
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0
  br label %bb3
bb3:
  %v17 = add i32 %v15, %v16
  %v18 = zext i32 %v17 to i64
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb4
bb4:
  %v20 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb5
bb5:
  %v21 = mul i32 %v19, %v20
  %v22 = zext i32 %v21 to i64
  %v23 = zext i32 %v11 to i64
  %v24 = insertvalue { i64, i64 } undef, i64 %v18, 0
  %v25 = insertvalue { i64, i64 } %v24, i64 %v23, 1
  %v26 = extractvalue { i64, i64 } %v25, 0
  %v27 = extractvalue { i64, i64 } %v25, 1
  %v28 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsgBauY1x2eDL_17infers_kernel_lib(i64 %v26, i64 %v27, i64 %v22) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v28, ptr %v12, align 8
  br label %bb12
bb6:
  %v29 = phi i64 [ %v73, %bb11 ], [ %v59, %bb12 ]
  %v30 = phi i64 [ %v74, %bb11 ], [ %v62, %bb12 ]
  %v31 = add i64 %v64, 1
  %v32 = icmp eq i64 %v31, 0
  %v33 = select i1 %v32, i8 0, i8 1
  %v34 = insertvalue { i8, { { i64 } } } undef, i8 %v33, 0
  %v35 = insertvalue { i8, { { i64 } } } %v34, i64 %v31, 1, 0, 0
  %v36 = extractvalue { i8, { { i64 } } } %v35, 0
  %v37 = zext i8 %v36 to i64
  %v38 = icmp eq i64 %v37, 1
  %v39 = extractvalue { i8, { { i64 } } } %v35, 1
  %v40 = alloca { { i64 } }, align 8
  store { { i64 } } %v39, ptr %v40, align 8
  %v41 = load i64, ptr %v40, align 8
  %v42 = icmp ugt i64 %v30, 0
  %v43 = xor i1 %v42, 1
  br i1 %v43, label %bb14, label %bb13
bb7:
  unreachable
bb8:
  %v44 = extractvalue { i64, i64 } %v72, 1
  %v45 = extractvalue { ptr, i64 } %v9, 1
  %v46 = icmp ult i64 %v44, %v45
  br i1 %v46, label %bb10, label %bb17
bb9:
  ret void
bb10:
  %v47 = extractvalue { ptr, i64 } %v9, 0
  %v48 = getelementptr inbounds i8, ptr %v47, i64 %v44
  %v49 = load i8, ptr %v48, align 1
  %v50 = call float @_infers_kernel_lib__shared__Fp8E5M2_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v49) #0
  br label %bb11
bb11:
  %v51 = bitcast float %v50 to i32
  %v52 = and i32 16, 31
  %v53 = lshr i32 %v51, %v52
  %v54 = trunc i32 %v53 to i16
  %v55 = extractvalue { ptr, i64 } %v10, 0
  %v56 = getelementptr inbounds i16, ptr %v55, i64 %v44
  store i16 %v54, ptr %v56, align 2
  br label %bb6
bb12:
  %v57 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v58 = getelementptr inbounds { i64, i64 }, ptr %v57, i32 0, i32 0
  %v59 = load i64, ptr %v58, align 8
  %v60 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 0
  %v61 = getelementptr inbounds { i64, i64 }, ptr %v60, i32 0, i32 1
  %v62 = load i64, ptr %v61, align 8
  %v63 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 1
  %v64 = load i64, ptr %v63, align 8
  %v65 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v12, i32 0, i32 2
  %v66 = load i1, ptr %v65, align 1
  br label %bb6
bb13:
  %v67 = add i64 %v29, %v41
  %v68 = sub i64 %v30, 1
  %v69 = insertvalue { i64, i64 } undef, i64 1, 0
  %v70 = insertvalue { i64, i64 } %v69, i64 %v29, 1
  br label %bb15
bb14:
  %v71 = insertvalue { i64, i64 } undef, i64 0, 0
  br label %bb15
bb15:
  %v72 = phi { i64, i64 } [ %v70, %bb13 ], [ %v71, %bb14 ]
  %v73 = phi i64 [ %v67, %bb13 ], [ %v29, %bb14 ]
  %v74 = phi i64 [ %v68, %bb13 ], [ %v30, %bb14 ]
  %v75 = extractvalue { i64, i64 } %v72, 0
  %v76 = bitcast i64 %v75 to i64
  %v77 = icmp eq i64 %v76, 0
  br i1 %v77, label %bb9, label %bb16
bb16:
  %v78 = icmp eq i64 %v76, 1
  br i1 %v78, label %bb8, label %bb7
bb17:
  unreachable
}

define float @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi i8 [ %v0, %entry ]
  %v2 = trunc i32 7 to i8
  %v3 = and i8 %v2, 7
  %v4 = lshr i8 %v1, %v3
  %v5 = and i8 %v4, 1
  %v6 = trunc i32 3 to i8
  %v7 = and i8 %v6, 7
  %v8 = lshr i8 %v1, %v7
  %v9 = and i8 %v8, 15
  %v10 = and i8 %v1, 7
  %v11 = icmp eq i8 %v9, 15
  br i1 %v11, label %bb1, label %bb2
bb1:
  br label %bb12
bb2:
  %v12 = icmp eq i8 %v9, 0
  %v13 = icmp eq i8 %v9, 0
  br i1 %v13, label %bb3, label %bb8
bb3:
  %v14 = icmp eq i8 %v10, 0
  br i1 %v14, label %bb4, label %bb8
bb4:
  %v15 = icmp eq i8 %v5, 0
  br i1 %v15, label %bb6, label %bb5
bb5:
  br label %bb7
bb6:
  br label %bb7
bb7:
  %v16 = phi float [ -0.0, %bb5 ], [ 0.0, %bb6 ]
  br label %bb12
bb8:
  %v17 = xor i1 %v12, 1
  br i1 %v17, label %bb10, label %bb9
bb9:
  br label %bb11
bb10:
  %v18 = zext i8 %v9 to i32
  %v19 = add i32 %v18, 120
  br label %bb11
bb11:
  %v20 = phi i32 [ 0, %bb9 ], [ %v19, %bb10 ]
  %v21 = zext i8 %v10 to i32
  %v22 = and i32 20, 31
  %v23 = shl i32 %v21, %v22
  %v24 = zext i8 %v5 to i32
  %v25 = and i32 31, 31
  %v26 = shl i32 %v24, %v25
  %v27 = and i32 23, 31
  %v28 = shl i32 %v20, %v27
  %v29 = or i32 %v26, %v28
  %v30 = or i32 %v29, %v23
  %v31 = bitcast i32 %v30 to float
  br label %bb12
bb12:
  %v32 = phi float [ 0x7FF8000000000000, %bb1 ], [ %v16, %bb7 ], [ %v31, %bb11 ]
  ret float %v32
}

define float @fp4_e2m1_to_f32(i8 %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi i8 [ %v0, %entry ]
  %v2 = trunc i32 3 to i8
  %v3 = and i8 %v2, 7
  %v4 = lshr i8 %v1, %v3
  %v5 = and i8 %v4, 1
  %v6 = and i8 %v1, 7
  %v7 = icmp eq i8 %v6, 0
  br i1 %v7, label %bb16, label %bb1
bb1:
  %v8 = icmp eq i8 %v6, 1
  br i1 %v8, label %bb15, label %bb2
bb2:
  %v9 = icmp eq i8 %v6, 2
  br i1 %v9, label %bb14, label %bb3
bb3:
  %v10 = icmp eq i8 %v6, 3
  br i1 %v10, label %bb13, label %bb4
bb4:
  %v11 = icmp eq i8 %v6, 4
  br i1 %v11, label %bb12, label %bb5
bb5:
  %v12 = icmp eq i8 %v6, 5
  br i1 %v12, label %bb11, label %bb6
bb6:
  %v13 = icmp eq i8 %v6, 6
  br i1 %v13, label %bb10, label %bb7
bb7:
  %v14 = icmp eq i8 %v6, 7
  br i1 %v14, label %bb9, label %bb8
bb8:
  unreachable
bb9:
  br label %bb17
bb10:
  br label %bb17
bb11:
  br label %bb17
bb12:
  br label %bb17
bb13:
  br label %bb17
bb14:
  br label %bb17
bb15:
  br label %bb17
bb16:
  br label %bb17
bb17:
  %v15 = phi float [ 6.0, %bb9 ], [ 4.0, %bb10 ], [ 3.0, %bb11 ], [ 2.0, %bb12 ], [ 1.5, %bb13 ], [ 1.0, %bb14 ], [ 0.5, %bb15 ], [ 0.0, %bb16 ]
  %v16 = icmp eq i8 %v5, 0
  br i1 %v16, label %bb19, label %bb18
bb18:
  %v17 = fneg float %v15
  br label %bb20
bb19:
  br label %bb20
bb20:
  %v18 = phi float [ %v17, %bb18 ], [ %v15, %bb19 ]
  ret float %v18
}

define { { i32, i32 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangelEE3newCsgBauY1x2eDL_17infers_kernel_lib(i32 %v0, i32 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i32, i32 } undef, i32 %v0, 0
  %v4 = insertvalue { i32, i32 } %v3, i32 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i32, i32 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  %v7 = icmp eq i64 %v6, 0
  br i1 %v7, label %bb2, label %bb1
bb1:
  %v8 = extractvalue { i32, i32 } %v5, 0
  %v9 = extractvalue { i32, i32 } %v5, 1
  %v10 = call { i32, i32 } @_RNvXs4_NtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtNtNtBb_3ops5range5RangelEINtB5_14SpecRangeSetupBQ_E5setupCsgBauY1x2eDL_17infers_kernel_lib(i32 %v8, i32 %v9, i64 %v6) #0
  br label %bb3
bb2:
  unreachable
bb3:
  %v11 = sub i64 %v6, 1
  %v12 = insertvalue { { i32, i32 }, i64, i1, [7 x i8] } undef, { i32, i32 } %v10, 0
  %v13 = insertvalue { { i32, i32 }, i64, i1, [7 x i8] } %v12, i64 %v11, 1
  %v14 = insertvalue { { i32, i32 }, i64, i1, [7 x i8] } %v13, i1 1, 2
  ret { { i32, i32 }, i64, i1, [7 x i8] } %v14
bb4:
  unreachable
bb5:
  unreachable
bb6:
  unreachable
}

define float @_infers_kernel_lib__shared__AutoRound_as_infers_kernel_lib__shared__Dequantize___dequant(i8 %v0, i8 %v1, float %v2) #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i8 [ %v0, %entry ]
  %v4 = phi i8 [ %v1, %entry ]
  %v5 = phi float [ %v2, %entry ]
  %v6 = add i8 %v4, 1
  %v7 = sub i8 %v3, %v6
  %v8 = sitofp i8 %v7 to float
  %v9 = fmul contract float %v8, %v5
  ret float %v9
}

define { i32, i32 } @_RNvXs3_NtNtCsiQ4CSjCKWVc_4core4iter5rangeINtNtNtB9_3ops5range5RangelENtB5_17RangeIteratorImpl8spec_nthCsgBauY1x2eDL_17infers_kernel_lib(ptr %v0, i64 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = alloca i32, align 4
  %v5 = getelementptr inbounds { i32, i32 }, ptr %v2, i32 0, i32 0
  %v6 = load i32, ptr %v5, align 4
  %v7 = call { i32, i32 } @_i32_as_std__iter__Step___forward_checked(i32 %v6, i64 %v3) #0
  br label %bb1
bb1:
  %v8 = extractvalue { i32, i32 } %v7, 0
  %v9 = zext i32 %v8 to i64
  %v10 = icmp eq i64 %v9, 1
  br i1 %v10, label %bb3, label %bb2
bb2:
  %v11 = icmp eq i64 %v9, 0
  br i1 %v11, label %bb8, label %bb11
bb3:
  %v12 = extractvalue { i32, i32 } %v7, 1
  store i32 %v12, ptr %v4, align 4
  %v13 = bitcast ptr %v4 to ptr
  %v14 = getelementptr inbounds { i32, i32 }, ptr %v2, i32 0, i32 1
  %v15 = call i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(ptr %v13, ptr %v14) #0
  br label %bb4
bb4:
  %v16 = xor i1 %v15, 1
  br i1 %v16, label %bb7, label %bb5
bb5:
  %v17 = load i32, ptr %v4, align 4
  %v18 = call i32 @_i32_as_std__iter__Step___forward_unchecked(i32 %v17, i64 1) #0
  br label %bb6
bb6:
  %v19 = getelementptr inbounds { i32, i32 }, ptr %v2, i32 0, i32 0
  store i32 %v18, ptr %v19, align 4
  %v20 = load i32, ptr %v4, align 4
  %v21 = insertvalue { i32, i32 } undef, i32 1, 0
  %v22 = insertvalue { i32, i32 } %v21, i32 %v20, 1
  br label %bb10
bb7:
  br label %bb9
bb8:
  br label %bb9
bb9:
  %v23 = getelementptr inbounds { i32, i32 }, ptr %v2, i32 0, i32 1
  %v24 = load i32, ptr %v23, align 4
  %v25 = getelementptr inbounds { i32, i32 }, ptr %v2, i32 0, i32 0
  store i32 %v24, ptr %v25, align 4
  %v26 = insertvalue { i32, i32 } undef, i32 0, 0
  br label %bb10
bb10:
  %v27 = phi { i32, i32 } [ %v22, %bb6 ], [ %v26, %bb9 ]
  ret { i32, i32 } %v27
bb11:
  unreachable
}

define { i64, i64 } @_std__ops__Range_usize__as_std__iter__adapters__step_by__SpecRangeSetup_std__ops__Range_usize_____setup(i64 %v0, i64 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i64, i64 } undef, i64 %v0, 0
  %v4 = insertvalue { i64, i64 } %v3, i64 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i64, i64 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  %v7 = alloca { i64, i64 }, align 8
  store { i64, i64 } %v5, ptr %v7, align 8
  %v8 = getelementptr inbounds { i64, i64 }, ptr %v7, i32 0, i32 0
  %v9 = load i64, ptr %v8, align 8
  %v10 = getelementptr inbounds { i64, i64 }, ptr %v7, i32 0, i32 1
  %v11 = load i64, ptr %v10, align 8
  %v12 = icmp ult i64 %v9, %v11
  %v13 = xor i1 %v12, 1
  br i1 %v13, label %bb2, label %bb1
bb1:
  %v14 = getelementptr inbounds { i64, i64 }, ptr %v7, i32 0, i32 0
  %v15 = load i64, ptr %v14, align 8
  %v16 = getelementptr inbounds { i64, i64 }, ptr %v7, i32 0, i32 1
  %v17 = load i64, ptr %v16, align 8
  %v18 = icmp ule i64 %v15, %v17
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb5, label %bb4
bb2:
  br label %bb3
bb3:
  %v20 = phi i64 [ 0, %bb2 ], [ %v24, %bb6 ]
  %v21 = icmp eq i64 %v6, 0
  %v22 = xor i1 %v21, 1
  br i1 %v22, label %bb7, label %bb11
bb4:
  %v23 = sub i64 %v17, %v15
  br label %bb6
bb5:
  br label %bb6
bb6:
  %v24 = phi i64 [ %v23, %bb4 ], [ 0, %bb5 ]
  br label %bb3
bb7:
  %v25 = udiv i64 %v20, %v6
  %v26 = urem i64 %v20, %v6
  %v27 = icmp ugt i64 %v26, 0
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb9, label %bb8
bb8:
  %v29 = add i64 %v25, 1
  br label %bb10
bb9:
  br label %bb10
bb10:
  %v30 = phi i64 [ %v29, %bb8 ], [ %v25, %bb9 ]
  %v31 = getelementptr inbounds { i64, i64 }, ptr %v7, i32 0, i32 1
  store i64 %v30, ptr %v31, align 8
  %v32 = load { i64, i64 }, ptr %v7, align 8
  ret { i64, i64 } %v32
bb11:
  unreachable
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_usize___lt(ptr %v0, ptr %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load i64, ptr %v2, align 8
  %v5 = load i64, ptr %v3, align 8
  %v6 = icmp ult i64 %v4, %v5
  ret i1 %v6
}

define float @_infers_kernel_lib__shared__Gguf_as_infers_kernel_lib__shared__Dequantize___dequant(i8 %v0, i8 %v1, float %v2) #0 {
entry:
  br label %bb0
bb0:
  %v3 = phi i8 [ %v0, %entry ]
  %v4 = phi i8 [ %v1, %entry ]
  %v5 = phi float [ %v2, %entry ]
  %v6 = sub i8 %v3, %v4
  %v7 = sitofp i8 %v6 to float
  %v8 = fmul contract float %v7, %v5
  ret float %v8
}

define i64 @_usize_as_std__iter__Step___forward_unchecked(i64 %v0, i64 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i64 [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = xor i1 0, 1
  br i1 %v4, label %bb2, label %bb1
bb1:
  br label %bb2
bb2:
  %v5 = add i64 %v2, %v3
  ret i64 %v5
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_usize___le(ptr %v0, ptr %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load i64, ptr %v2, align 8
  %v5 = load i64, ptr %v3, align 8
  %v6 = icmp ule i64 %v4, %v5
  ret i1 %v6
}

define i8 @_infers_kernel_lib__shared__Fp8E5M2_as_infers_kernel_lib__shared__Fp8Format___quantize(float %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi float [ %v0, %entry ]
  %v2 = bitcast float %v1 to i32
  %v3 = and i32 31, 31
  %v4 = lshr i32 %v2, %v3
  %v5 = and i32 %v4, 1
  %v6 = and i32 23, 31
  %v7 = lshr i32 %v2, %v6
  %v8 = and i32 %v7, 255
  %v9 = and i32 %v2, 8388607
  %v10 = icmp eq i32 %v8, 255
  br i1 %v10, label %bb1, label %bb10
bb1:
  %v11 = icmp eq i32 %v9, 0
  br i1 %v11, label %bb6, label %bb2
bb2:
  %v12 = icmp eq i32 %v5, 0
  br i1 %v12, label %bb3, label %bb4
bb3:
  br label %bb5
bb4:
  br label %bb5
bb5:
  %v13 = phi i8 [ 127, %bb3 ], [ 255, %bb4 ]
  br label %bb21
bb6:
  %v14 = icmp eq i32 %v5, 0
  br i1 %v14, label %bb7, label %bb8
bb7:
  br label %bb9
bb8:
  br label %bb9
bb9:
  %v15 = phi i8 [ 124, %bb7 ], [ 252, %bb8 ]
  br label %bb21
bb10:
  %v16 = icmp eq i32 %v8, 0
  br i1 %v16, label %bb11, label %bb13
bb11:
  %v17 = icmp eq i32 %v9, 0
  br i1 %v17, label %bb12, label %bb13
bb12:
  %v18 = and i32 %v5, 1
  %v19 = trunc i32 %v18 to i8
  %v20 = mul i8 %v19, 128
  br label %bb22
bb13:
  %v21 = bitcast i32 %v8 to i32
  %v22 = sub i32 %v21, 127
  %v23 = add i32 %v22, 15
  %v24 = icmp sge i32 %v23, 31
  %v25 = xor i1 %v24, 1
  br i1 %v25, label %bb15, label %bb14
bb14:
  %v26 = icmp eq i32 %v5, 0
  br i1 %v26, label %bb17, label %bb16
bb15:
  %v27 = icmp slt i32 %v23, 0
  %v28 = xor i1 %v27, 1
  br i1 %v28, label %bb20, label %bb19
bb16:
  br label %bb18
bb17:
  br label %bb18
bb18:
  %v29 = phi i8 [ 251, %bb16 ], [ 123, %bb17 ]
  br label %bb22
bb19:
  %v30 = and i32 %v5, 1
  %v31 = trunc i32 %v30 to i8
  %v32 = mul i8 %v31, 128
  br label %bb22
bb20:
  %v33 = and i32 21, 31
  %v34 = lshr i32 %v9, %v33
  %v35 = and i32 %v34, 3
  %v36 = trunc i32 %v35 to i8
  %v37 = and i32 %v5, 1
  %v38 = trunc i32 %v37 to i8
  %v39 = trunc i32 7 to i8
  %v40 = and i8 %v39, 7
  %v41 = shl i8 %v38, %v40
  %v42 = trunc i32 %v23 to i8
  %v43 = trunc i32 2 to i8
  %v44 = and i8 %v43, 7
  %v45 = shl i8 %v42, %v44
  %v46 = or i8 %v41, %v45
  %v47 = or i8 %v46, %v36
  br label %bb22
bb21:
  %v48 = phi i8 [ %v13, %bb5 ], [ %v15, %bb9 ]
  br label %bb22
bb22:
  %v49 = phi i8 [ %v20, %bb12 ], [ %v29, %bb18 ], [ %v32, %bb19 ], [ %v47, %bb20 ], [ %v48, %bb21 ]
  ret i8 %v49
}

define i8 @_infers_kernel_lib__shared__Fp8E4M3_as_infers_kernel_lib__shared__Fp8Format___quantize(float %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi float [ %v0, %entry ]
  %v2 = bitcast float %v1 to i32
  %v3 = and i32 31, 31
  %v4 = lshr i32 %v2, %v3
  %v5 = and i32 %v4, 1
  %v6 = and i32 23, 31
  %v7 = lshr i32 %v2, %v6
  %v8 = and i32 %v7, 255
  %v9 = and i32 %v2, 8388607
  %v10 = icmp eq i32 %v8, 255
  br i1 %v10, label %bb1, label %bb7
bb1:
  %v11 = icmp eq i32 %v9, 0
  br i1 %v11, label %bb3, label %bb2
bb2:
  br label %bb18
bb3:
  %v12 = icmp eq i32 %v5, 0
  br i1 %v12, label %bb4, label %bb5
bb4:
  br label %bb6
bb5:
  br label %bb6
bb6:
  %v13 = phi i8 [ 119, %bb4 ], [ 247, %bb5 ]
  br label %bb18
bb7:
  %v14 = icmp eq i32 %v8, 0
  br i1 %v14, label %bb8, label %bb10
bb8:
  %v15 = icmp eq i32 %v9, 0
  br i1 %v15, label %bb9, label %bb10
bb9:
  %v16 = and i32 %v5, 1
  %v17 = trunc i32 %v16 to i8
  %v18 = mul i8 %v17, 128
  br label %bb19
bb10:
  %v19 = bitcast i32 %v8 to i32
  %v20 = sub i32 %v19, 127
  %v21 = add i32 %v20, 7
  %v22 = icmp sge i32 %v21, 15
  %v23 = xor i1 %v22, 1
  br i1 %v23, label %bb12, label %bb11
bb11:
  %v24 = icmp eq i32 %v5, 0
  br i1 %v24, label %bb14, label %bb13
bb12:
  %v25 = icmp slt i32 %v21, 0
  %v26 = xor i1 %v25, 1
  br i1 %v26, label %bb17, label %bb16
bb13:
  br label %bb15
bb14:
  br label %bb15
bb15:
  %v27 = phi i8 [ 247, %bb13 ], [ 119, %bb14 ]
  br label %bb19
bb16:
  %v28 = and i32 %v5, 1
  %v29 = trunc i32 %v28 to i8
  %v30 = mul i8 %v29, 128
  br label %bb19
bb17:
  %v31 = and i32 20, 31
  %v32 = lshr i32 %v9, %v31
  %v33 = and i32 %v32, 7
  %v34 = trunc i32 %v33 to i8
  %v35 = and i32 %v5, 1
  %v36 = trunc i32 %v35 to i8
  %v37 = trunc i32 7 to i8
  %v38 = and i8 %v37, 7
  %v39 = shl i8 %v36, %v38
  %v40 = trunc i32 %v21 to i8
  %v41 = trunc i32 3 to i8
  %v42 = and i8 %v41, 7
  %v43 = shl i8 %v40, %v42
  %v44 = or i8 %v39, %v43
  %v45 = or i8 %v44, %v34
  br label %bb19
bb18:
  %v46 = phi i8 [ 127, %bb2 ], [ %v13, %bb6 ]
  br label %bb19
bb19:
  %v47 = phi i8 [ %v18, %bb9 ], [ %v27, %bb15 ], [ %v30, %bb16 ], [ %v45, %bb17 ], [ %v46, %bb18 ]
  ret i8 %v47
}

define float @_infers_kernel_lib__shared__Fp8E5M2_as_infers_kernel_lib__shared__Fp8Format___dequantize(i8 %v0) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v1 = phi i8 [ %v0, %entry ]
  %v2 = trunc i32 7 to i8
  %v3 = and i8 %v2, 7
  %v4 = lshr i8 %v1, %v3
  %v5 = and i8 %v4, 1
  %v6 = trunc i32 2 to i8
  %v7 = and i8 %v6, 7
  %v8 = lshr i8 %v1, %v7
  %v9 = and i8 %v8, 31
  %v10 = and i8 %v1, 3
  %v11 = icmp eq i8 %v9, 31
  br i1 %v11, label %bb1, label %bb2
bb1:
  br label %bb12
bb2:
  %v12 = icmp eq i8 %v9, 0
  %v13 = icmp eq i8 %v9, 0
  br i1 %v13, label %bb3, label %bb8
bb3:
  %v14 = icmp eq i8 %v10, 0
  br i1 %v14, label %bb4, label %bb8
bb4:
  %v15 = icmp eq i8 %v5, 0
  br i1 %v15, label %bb6, label %bb5
bb5:
  br label %bb7
bb6:
  br label %bb7
bb7:
  %v16 = phi float [ -0.0, %bb5 ], [ 0.0, %bb6 ]
  br label %bb12
bb8:
  %v17 = xor i1 %v12, 1
  br i1 %v17, label %bb10, label %bb9
bb9:
  br label %bb11
bb10:
  %v18 = zext i8 %v9 to i32
  %v19 = add i32 %v18, 112
  br label %bb11
bb11:
  %v20 = phi i32 [ 0, %bb9 ], [ %v19, %bb10 ]
  %v21 = zext i8 %v10 to i32
  %v22 = and i32 21, 31
  %v23 = shl i32 %v21, %v22
  %v24 = zext i8 %v5 to i32
  %v25 = and i32 31, 31
  %v26 = shl i32 %v24, %v25
  %v27 = and i32 23, 31
  %v28 = shl i32 %v20, %v27
  %v29 = or i32 %v26, %v28
  %v30 = or i32 %v29, %v23
  %v31 = bitcast i32 %v30 to float
  br label %bb12
bb12:
  %v32 = phi float [ 0x7FF8000000000000, %bb1 ], [ %v16, %bb7 ], [ %v31, %bb11 ]
  ret float %v32
}

define { i32, i32 } @_RNvXs4_NtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtNtNtBb_3ops5range5RangelEINtB5_14SpecRangeSetupBQ_E5setupCsgBauY1x2eDL_17infers_kernel_lib(i32 %v0, i32 %v1, i64 %v2) #0 {
entry:
  %v3 = insertvalue { i32, i32 } undef, i32 %v0, 0
  %v4 = insertvalue { i32, i32 } %v3, i32 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { i32, i32 } [ %v4, %entry ]
  %v6 = phi i64 [ %v2, %entry ]
  ret { i32, i32 } %v5
}

define { i32, i32 } @_i32_as_std__iter__Step___forward_checked(i32 %v0, i64 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = icmp ugt i64 %v3, 4294967295
  %v5 = xor i1 %v4, 1
  br i1 %v5, label %bb6, label %bb5
bb1:
  %v6 = insertvalue { i32, i32 } undef, i32 1, 0
  %v7 = insertvalue { i32, i32 } %v6, i32 %v17, 1
  br label %bb3
bb2:
  %v8 = insertvalue { i32, i32 } undef, i32 0, 0
  br label %bb3
bb3:
  %v9 = phi { i32, i32 } [ %v7, %bb1 ], [ %v8, %bb2 ]
  br label %bb4
bb4:
  %v10 = phi { i32, i32 } [ %v9, %bb3 ], [ %v11, %bb5 ]
  ret { i32, i32 } %v10
bb5:
  %v11 = insertvalue { i32, i32 } undef, i32 0, 0
  br label %bb4
bb6:
  %v12 = trunc i64 %v3 to i32
  %v13 = insertvalue { i32, i32 } undef, i32 0, 0
  %v14 = insertvalue { i32, i32 } %v13, i32 %v12, 1
  %v15 = extractvalue { i32, i32 } %v14, 1
  %v16 = bitcast i32 %v15 to i32
  %v17 = add i32 %v2, %v16
  %v18 = icmp sge i32 %v17, %v2
  %v19 = xor i1 %v18, 1
  br i1 %v19, label %bb2, label %bb1
}

define i1 @std__cmp__impls___impl_std__cmp__PartialOrd_for_i32___lt(ptr %v0, ptr %v1) alwaysinline #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi ptr [ %v0, %entry ]
  %v3 = phi ptr [ %v1, %entry ]
  %v4 = load i32, ptr %v2, align 4
  %v5 = load i32, ptr %v3, align 4
  %v6 = icmp slt i32 %v4, %v5
  ret i1 %v6
}

define i32 @_i32_as_std__iter__Step___forward_unchecked(i32 %v0, i64 %v1) #0 {
entry:
  br label %bb0
bb0:
  %v2 = phi i32 [ %v0, %entry ]
  %v3 = phi i64 [ %v1, %entry ]
  %v4 = trunc i64 %v3 to i32
  %v5 = bitcast i32 %v4 to i32
  %v6 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 %v2, i32 %v5) #0
  %v7 = extractvalue { i32, i1 } %v6, 0
  %v8 = extractvalue { i32, i1 } %v6, 1
  %v9 = icmp slt i32 %v5, 0
  %v10 = xor i1 %v8, %v9
  %v11 = xor i1 %v10, 1
  br i1 %v11, label %bb3, label %bb1
bb1:
  br label %bb2
bb2:
  %v12 = insertvalue { i32, i32 } undef, i32 0, 0
  br label %bb4
bb3:
  %v13 = insertvalue { i32, i32 } undef, i32 1, 0
  %v14 = insertvalue { i32, i32 } %v13, i32 %v7, 1
  %v15 = extractvalue { i32, i32 } %v14, 1
  ret i32 %v15
bb4:
  unreachable
}


@llvm.used = appending global [42 x ptr] [ptr @int4_gemm_auto_round, ptr @int4_gemm_v4_ksplit, ptr @int4_gemm_warp_split, ptr @int4_gemm_v3_ksplit_sm, ptr @int4_gemm_auto_round_tiled, ptr @reduce_partial_sums_bf16, ptr @int4_gemm_warp, ptr @int4_gemm_gguf, ptr @int4_dequant_to_bf16, ptr @int4_gemm_auto_round_ksplit, ptr @infers_gdn_chunked_gated_delta_prefill_bf16, ptr @infers_gdn_mamba2_update_bf16, ptr @infers_gdn_gated_delta_prefill_bf16, ptr @infers_gdn_update_bf16, ptr @infers_gdn_recurrent_step_bf16, ptr @infers_gdn_gated_delta_update_bf16, ptr @infers_rope_bf16, ptr @infers_paged_kv_write_bf16, ptr @infers_paged_attention_decode_bf16, ptr @infers_paged_kv_read_bf16, ptr @sanitize_nan_bf16, ptr @infers_kv_cache_write_bf16, ptr @infers_softmax_bf16, ptr @infers_embedding_gather_bf16, ptr @infers_argmax_bf16, ptr @infers_add_bf16, ptr @infers_attn_output_gate_bf16, ptr @infers_silu_bf16, ptr @infers_conv1d_depthwise_silu_bf16, ptr @infers_silu_glu_bf16, ptr @infers_rmsnorm_bf16, ptr @infers_l2norm_bf16, ptr @infers_rms_norm_gated_bf16, ptr @infers_fp8_dequantize_e4m3, ptr @infers_fp8_quantize_e5m2, ptr @infers_fp8_quantize_e4m3, ptr @infers_fp8_dequantize_e5m2, ptr @nvfp4_gemm_v3_ksplit, ptr @nvfp4_dequant_to_bf16, ptr @nvfp4_gemm_fused, ptr @nvfp4_gemm_fused_ksplit, ptr @bf16_gemm_tiled], section "llvm.metadata"

attributes #0 = { convergent }

!0 = !{ptr @int4_gemm_auto_round, !"kernel", i32 1}
!1 = !{ptr @int4_gemm_gguf, !"kernel", i32 1}
!2 = !{ptr @int4_dequant_to_bf16, !"kernel", i32 1}
!3 = !{ptr @sanitize_nan_bf16, !"kernel", i32 1}
!4 = !{ptr @nvfp4_dequant_to_bf16, !"kernel", i32 1}
!5 = !{ptr @nvfp4_gemm_fused, !"kernel", i32 1}
!6 = !{ptr @int4_gemm_v4_ksplit, !"kernel", i32 1}
!7 = !{ptr @int4_gemm_v4_ksplit, !"maxntidx", i32 64}
!8 = !{ptr @int4_gemm_v4_ksplit, !"maxntidy", i32 1}
!9 = !{ptr @int4_gemm_v4_ksplit, !"maxntidz", i32 1}
!10 = !{ptr @int4_gemm_warp_split, !"kernel", i32 1}
!11 = !{ptr @int4_gemm_warp_split, !"maxntidx", i32 256}
!12 = !{ptr @int4_gemm_warp_split, !"maxntidy", i32 1}
!13 = !{ptr @int4_gemm_warp_split, !"maxntidz", i32 1}
!14 = !{ptr @int4_gemm_v3_ksplit_sm, !"kernel", i32 1}
!15 = !{ptr @int4_gemm_v3_ksplit_sm, !"maxntidx", i32 64}
!16 = !{ptr @int4_gemm_v3_ksplit_sm, !"maxntidy", i32 1}
!17 = !{ptr @int4_gemm_v3_ksplit_sm, !"maxntidz", i32 1}
!18 = !{ptr @int4_gemm_auto_round_tiled, !"kernel", i32 1}
!19 = !{ptr @int4_gemm_auto_round_tiled, !"maxntidx", i32 64}
!20 = !{ptr @int4_gemm_auto_round_tiled, !"maxntidy", i32 1}
!21 = !{ptr @int4_gemm_auto_round_tiled, !"maxntidz", i32 1}
!22 = !{ptr @reduce_partial_sums_bf16, !"kernel", i32 1}
!23 = !{ptr @reduce_partial_sums_bf16, !"maxntidx", i32 64}
!24 = !{ptr @reduce_partial_sums_bf16, !"maxntidy", i32 1}
!25 = !{ptr @reduce_partial_sums_bf16, !"maxntidz", i32 1}
!26 = !{ptr @int4_gemm_warp, !"kernel", i32 1}
!27 = !{ptr @int4_gemm_warp, !"maxntidx", i32 256}
!28 = !{ptr @int4_gemm_warp, !"maxntidy", i32 1}
!29 = !{ptr @int4_gemm_warp, !"maxntidz", i32 1}
!30 = !{ptr @int4_gemm_auto_round_ksplit, !"kernel", i32 1}
!31 = !{ptr @int4_gemm_auto_round_ksplit, !"maxntidx", i32 64}
!32 = !{ptr @int4_gemm_auto_round_ksplit, !"maxntidy", i32 1}
!33 = !{ptr @int4_gemm_auto_round_ksplit, !"maxntidz", i32 1}
!34 = !{ptr @infers_gdn_chunked_gated_delta_prefill_bf16, !"kernel", i32 1}
!35 = !{ptr @infers_gdn_chunked_gated_delta_prefill_bf16, !"maxntidx", i32 256}
!36 = !{ptr @infers_gdn_chunked_gated_delta_prefill_bf16, !"maxntidy", i32 1}
!37 = !{ptr @infers_gdn_chunked_gated_delta_prefill_bf16, !"maxntidz", i32 1}
!38 = !{ptr @infers_gdn_mamba2_update_bf16, !"kernel", i32 1}
!39 = !{ptr @infers_gdn_mamba2_update_bf16, !"maxntidx", i32 256}
!40 = !{ptr @infers_gdn_mamba2_update_bf16, !"maxntidy", i32 1}
!41 = !{ptr @infers_gdn_mamba2_update_bf16, !"maxntidz", i32 1}
!42 = !{ptr @infers_gdn_gated_delta_prefill_bf16, !"kernel", i32 1}
!43 = !{ptr @infers_gdn_gated_delta_prefill_bf16, !"maxntidx", i32 256}
!44 = !{ptr @infers_gdn_gated_delta_prefill_bf16, !"maxntidy", i32 1}
!45 = !{ptr @infers_gdn_gated_delta_prefill_bf16, !"maxntidz", i32 1}
!46 = !{ptr @infers_gdn_update_bf16, !"kernel", i32 1}
!47 = !{ptr @infers_gdn_update_bf16, !"maxntidx", i32 256}
!48 = !{ptr @infers_gdn_update_bf16, !"maxntidy", i32 1}
!49 = !{ptr @infers_gdn_update_bf16, !"maxntidz", i32 1}
!50 = !{ptr @infers_gdn_recurrent_step_bf16, !"kernel", i32 1}
!51 = !{ptr @infers_gdn_recurrent_step_bf16, !"maxntidx", i32 256}
!52 = !{ptr @infers_gdn_recurrent_step_bf16, !"maxntidy", i32 1}
!53 = !{ptr @infers_gdn_recurrent_step_bf16, !"maxntidz", i32 1}
!54 = !{ptr @infers_gdn_gated_delta_update_bf16, !"kernel", i32 1}
!55 = !{ptr @infers_gdn_gated_delta_update_bf16, !"maxntidx", i32 256}
!56 = !{ptr @infers_gdn_gated_delta_update_bf16, !"maxntidy", i32 1}
!57 = !{ptr @infers_gdn_gated_delta_update_bf16, !"maxntidz", i32 1}
!58 = !{ptr @infers_rope_bf16, !"kernel", i32 1}
!59 = !{ptr @infers_rope_bf16, !"maxntidx", i32 256}
!60 = !{ptr @infers_rope_bf16, !"maxntidy", i32 1}
!61 = !{ptr @infers_rope_bf16, !"maxntidz", i32 1}
!62 = !{ptr @infers_paged_kv_write_bf16, !"kernel", i32 1}
!63 = !{ptr @infers_paged_kv_write_bf16, !"maxntidx", i32 256}
!64 = !{ptr @infers_paged_kv_write_bf16, !"maxntidy", i32 1}
!65 = !{ptr @infers_paged_kv_write_bf16, !"maxntidz", i32 1}
!66 = !{ptr @infers_paged_attention_decode_bf16, !"kernel", i32 1}
!67 = !{ptr @infers_paged_attention_decode_bf16, !"maxntidx", i32 256}
!68 = !{ptr @infers_paged_attention_decode_bf16, !"maxntidy", i32 1}
!69 = !{ptr @infers_paged_attention_decode_bf16, !"maxntidz", i32 1}
!70 = !{ptr @infers_paged_kv_read_bf16, !"kernel", i32 1}
!71 = !{ptr @infers_paged_kv_read_bf16, !"maxntidx", i32 256}
!72 = !{ptr @infers_paged_kv_read_bf16, !"maxntidy", i32 1}
!73 = !{ptr @infers_paged_kv_read_bf16, !"maxntidz", i32 1}
!74 = !{ptr @infers_kv_cache_write_bf16, !"kernel", i32 1}
!75 = !{ptr @infers_kv_cache_write_bf16, !"maxntidx", i32 256}
!76 = !{ptr @infers_kv_cache_write_bf16, !"maxntidy", i32 1}
!77 = !{ptr @infers_kv_cache_write_bf16, !"maxntidz", i32 1}
!78 = !{ptr @infers_softmax_bf16, !"kernel", i32 1}
!79 = !{ptr @infers_softmax_bf16, !"maxntidx", i32 256}
!80 = !{ptr @infers_softmax_bf16, !"maxntidy", i32 1}
!81 = !{ptr @infers_softmax_bf16, !"maxntidz", i32 1}
!82 = !{ptr @infers_embedding_gather_bf16, !"kernel", i32 1}
!83 = !{ptr @infers_embedding_gather_bf16, !"maxntidx", i32 256}
!84 = !{ptr @infers_embedding_gather_bf16, !"maxntidy", i32 1}
!85 = !{ptr @infers_embedding_gather_bf16, !"maxntidz", i32 1}
!86 = !{ptr @infers_argmax_bf16, !"kernel", i32 1}
!87 = !{ptr @infers_argmax_bf16, !"maxntidx", i32 256}
!88 = !{ptr @infers_argmax_bf16, !"maxntidy", i32 1}
!89 = !{ptr @infers_argmax_bf16, !"maxntidz", i32 1}
!90 = !{ptr @infers_add_bf16, !"kernel", i32 1}
!91 = !{ptr @infers_add_bf16, !"maxntidx", i32 256}
!92 = !{ptr @infers_add_bf16, !"maxntidy", i32 1}
!93 = !{ptr @infers_add_bf16, !"maxntidz", i32 1}
!94 = !{ptr @infers_attn_output_gate_bf16, !"kernel", i32 1}
!95 = !{ptr @infers_attn_output_gate_bf16, !"maxntidx", i32 256}
!96 = !{ptr @infers_attn_output_gate_bf16, !"maxntidy", i32 1}
!97 = !{ptr @infers_attn_output_gate_bf16, !"maxntidz", i32 1}
!98 = !{ptr @infers_silu_bf16, !"kernel", i32 1}
!99 = !{ptr @infers_silu_bf16, !"maxntidx", i32 256}
!100 = !{ptr @infers_silu_bf16, !"maxntidy", i32 1}
!101 = !{ptr @infers_silu_bf16, !"maxntidz", i32 1}
!102 = !{ptr @infers_conv1d_depthwise_silu_bf16, !"kernel", i32 1}
!103 = !{ptr @infers_conv1d_depthwise_silu_bf16, !"maxntidx", i32 256}
!104 = !{ptr @infers_conv1d_depthwise_silu_bf16, !"maxntidy", i32 1}
!105 = !{ptr @infers_conv1d_depthwise_silu_bf16, !"maxntidz", i32 1}
!106 = !{ptr @infers_silu_glu_bf16, !"kernel", i32 1}
!107 = !{ptr @infers_silu_glu_bf16, !"maxntidx", i32 256}
!108 = !{ptr @infers_silu_glu_bf16, !"maxntidy", i32 1}
!109 = !{ptr @infers_silu_glu_bf16, !"maxntidz", i32 1}
!110 = !{ptr @infers_rmsnorm_bf16, !"kernel", i32 1}
!111 = !{ptr @infers_rmsnorm_bf16, !"maxntidx", i32 256}
!112 = !{ptr @infers_rmsnorm_bf16, !"maxntidy", i32 1}
!113 = !{ptr @infers_rmsnorm_bf16, !"maxntidz", i32 1}
!114 = !{ptr @infers_l2norm_bf16, !"kernel", i32 1}
!115 = !{ptr @infers_l2norm_bf16, !"maxntidx", i32 256}
!116 = !{ptr @infers_l2norm_bf16, !"maxntidy", i32 1}
!117 = !{ptr @infers_l2norm_bf16, !"maxntidz", i32 1}
!118 = !{ptr @infers_rms_norm_gated_bf16, !"kernel", i32 1}
!119 = !{ptr @infers_rms_norm_gated_bf16, !"maxntidx", i32 256}
!120 = !{ptr @infers_rms_norm_gated_bf16, !"maxntidy", i32 1}
!121 = !{ptr @infers_rms_norm_gated_bf16, !"maxntidz", i32 1}
!122 = !{ptr @infers_fp8_dequantize_e4m3, !"kernel", i32 1}
!123 = !{ptr @infers_fp8_dequantize_e4m3, !"maxntidx", i32 256}
!124 = !{ptr @infers_fp8_dequantize_e4m3, !"maxntidy", i32 1}
!125 = !{ptr @infers_fp8_dequantize_e4m3, !"maxntidz", i32 1}
!126 = !{ptr @infers_fp8_quantize_e5m2, !"kernel", i32 1}
!127 = !{ptr @infers_fp8_quantize_e5m2, !"maxntidx", i32 256}
!128 = !{ptr @infers_fp8_quantize_e5m2, !"maxntidy", i32 1}
!129 = !{ptr @infers_fp8_quantize_e5m2, !"maxntidz", i32 1}
!130 = !{ptr @infers_fp8_quantize_e4m3, !"kernel", i32 1}
!131 = !{ptr @infers_fp8_quantize_e4m3, !"maxntidx", i32 256}
!132 = !{ptr @infers_fp8_quantize_e4m3, !"maxntidy", i32 1}
!133 = !{ptr @infers_fp8_quantize_e4m3, !"maxntidz", i32 1}
!134 = !{ptr @infers_fp8_dequantize_e5m2, !"kernel", i32 1}
!135 = !{ptr @infers_fp8_dequantize_e5m2, !"maxntidx", i32 256}
!136 = !{ptr @infers_fp8_dequantize_e5m2, !"maxntidy", i32 1}
!137 = !{ptr @infers_fp8_dequantize_e5m2, !"maxntidz", i32 1}
!138 = !{ptr @nvfp4_gemm_v3_ksplit, !"kernel", i32 1}
!139 = !{ptr @nvfp4_gemm_v3_ksplit, !"maxntidx", i32 64}
!140 = !{ptr @nvfp4_gemm_v3_ksplit, !"maxntidy", i32 1}
!141 = !{ptr @nvfp4_gemm_v3_ksplit, !"maxntidz", i32 1}
!142 = !{ptr @nvfp4_gemm_fused_ksplit, !"kernel", i32 1}
!143 = !{ptr @nvfp4_gemm_fused_ksplit, !"maxntidx", i32 64}
!144 = !{ptr @nvfp4_gemm_fused_ksplit, !"maxntidy", i32 1}
!145 = !{ptr @nvfp4_gemm_fused_ksplit, !"maxntidz", i32 1}
!146 = !{ptr @bf16_gemm_tiled, !"kernel", i32 1}
!147 = !{ptr @bf16_gemm_tiled, !"maxntidx", i32 256}
!148 = !{ptr @bf16_gemm_tiled, !"maxntidy", i32 1}
!149 = !{ptr @bf16_gemm_tiled, !"maxntidz", i32 1}
!nvvm.annotations = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13, !14, !15, !16, !17, !18, !19, !20, !21, !22, !23, !24, !25, !26, !27, !28, !29, !30, !31, !32, !33, !34, !35, !36, !37, !38, !39, !40, !41, !42, !43, !44, !45, !46, !47, !48, !49, !50, !51, !52, !53, !54, !55, !56, !57, !58, !59, !60, !61, !62, !63, !64, !65, !66, !67, !68, !69, !70, !71, !72, !73, !74, !75, !76, !77, !78, !79, !80, !81, !82, !83, !84, !85, !86, !87, !88, !89, !90, !91, !92, !93, !94, !95, !96, !97, !98, !99, !100, !101, !102, !103, !104, !105, !106, !107, !108, !109, !110, !111, !112, !113, !114, !115, !116, !117, !118, !119, !120, !121, !122, !123, !124, !125, !126, !127, !128, !129, !130, !131, !132, !133, !134, !135, !136, !137, !138, !139, !140, !141, !142, !143, !144, !145, !146, !147, !148, !149}

!nvvmir.version = !{!150}
!150 = !{i32 2, i32 0, i32 3, i32 2}
