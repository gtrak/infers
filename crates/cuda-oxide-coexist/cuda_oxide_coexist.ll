; ModuleID = 'builtin.module'
source_filename = "cuda_oxide_coexist"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

declare i32 @llvm.nvvm.read.ptx.sreg.ntid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.nctaid.x()

define ptx_kernel void @mem_copy(ptr %v0, i64 %v1, ptr %v2, i64 %v3, i32 %v4) #0 {
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
  %v14 = bitcast ptr %v12 to ptr
  %v15 = call i64 @cuda_device____internal__index_1d(ptr %v14) #0
  br label %bb1
bb1:
  %v16 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v17 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb3
bb3:
  %v18 = mul i32 %v16, %v17
  %v19 = zext i32 %v11 to i64
  %v20 = insertvalue { i64, i64 } undef, i64 %v15, 0
  %v21 = insertvalue { i64, i64 } %v20, i64 %v19, 1
  %v22 = zext i32 %v18 to i64
  %v23 = extractvalue { i64, i64 } %v21, 0
  %v24 = extractvalue { i64, i64 } %v21, 1
  %v25 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsb6GslpvRJ9R_18cuda_oxide_coexist(i64 %v23, i64 %v24, i64 %v22) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v25, ptr %v13, align 8
  br label %bb6
bb4:
  %v26 = phi i64 [ %v56, %bb5 ], [ %v48, %bb6 ]
  %v27 = phi i64 [ %v57, %bb5 ], [ %v51, %bb6 ]
  %v28 = add i64 %v53, 1
  %v29 = icmp eq i64 %v28, 0
  %v30 = select i1 %v29, i8 0, i8 1
  %v31 = insertvalue { i8, { { i64 } } } undef, i8 %v30, 0
  %v32 = insertvalue { i8, { { i64 } } } %v31, i64 %v28, 1, 0, 0
  %v33 = extractvalue { i8, { { i64 } } } %v32, 0
  %v34 = zext i8 %v33 to i64
  %v35 = icmp eq i64 %v34, 1
  %v36 = extractvalue { i8, { { i64 } } } %v32, 1
  %v37 = alloca { { i64 } }, align 8
  store { { i64 } } %v36, ptr %v37, align 8
  %v38 = load i64, ptr %v37, align 8
  %v39 = icmp ugt i64 %v27, 0
  %v40 = xor i1 %v39, 1
  br i1 %v40, label %bb8, label %bb7
bb5:
  %v41 = extractvalue { ptr, i64 } %v9, 0
  %v42 = getelementptr inbounds float, ptr %v41, i64 %v60
  %v43 = load float, ptr %v42, align 4
  %v44 = extractvalue { ptr, i64 } %v10, 0
  %v45 = getelementptr inbounds float, ptr %v44, i64 %v60
  store float %v43, ptr %v45, align 4
  br label %bb4
bb6:
  %v46 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 0
  %v47 = getelementptr inbounds { i64, i64 }, ptr %v46, i32 0, i32 0
  %v48 = load i64, ptr %v47, align 8
  %v49 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 0
  %v50 = getelementptr inbounds { i64, i64 }, ptr %v49, i32 0, i32 1
  %v51 = load i64, ptr %v50, align 8
  %v52 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 1
  %v53 = load i64, ptr %v52, align 8
  %v54 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v13, i32 0, i32 2
  %v55 = load i1, ptr %v54, align 1
  br label %bb4
bb7:
  %v56 = add i64 %v26, %v38
  %v57 = sub i64 %v27, 1
  %v58 = insertvalue { i64, i64 } undef, i64 1, 0
  %v59 = insertvalue { i64, i64 } %v58, i64 %v26, 1
  %v60 = extractvalue { i64, i64 } %v59, 1
  %v61 = extractvalue { ptr, i64 } %v9, 1
  %v62 = icmp ult i64 %v60, %v61
  br i1 %v62, label %bb5, label %bb9
bb8:
  ret void
bb9:
  unreachable
}

define ptx_kernel void @scalar_mul(ptr %v0, i64 %v1, ptr %v2, i64 %v3, float %v4, i32 %v5) #0 {
entry:
  %v6 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v7 = insertvalue { ptr, i64 } %v6, i64 %v1, 1
  %v8 = insertvalue { ptr, i64 } undef, ptr %v2, 0
  %v9 = insertvalue { ptr, i64 } %v8, i64 %v3, 1
  br label %bb0
bb0:
  %v10 = phi { ptr, i64 } [ %v7, %entry ]
  %v11 = phi { ptr, i64 } [ %v9, %entry ]
  %v12 = phi float [ %v4, %entry ]
  %v13 = phi i32 [ %v5, %entry ]
  %v14 = alloca {  }, align 1
  %v15 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v16 = bitcast ptr %v14 to ptr
  %v17 = call i64 @cuda_device____internal__index_1d(ptr %v16) #0
  br label %bb1
bb1:
  %v18 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v19 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb3
bb3:
  %v20 = mul i32 %v18, %v19
  %v21 = zext i32 %v13 to i64
  %v22 = insertvalue { i64, i64 } undef, i64 %v17, 0
  %v23 = insertvalue { i64, i64 } %v22, i64 %v21, 1
  %v24 = zext i32 %v20 to i64
  %v25 = extractvalue { i64, i64 } %v23, 0
  %v26 = extractvalue { i64, i64 } %v23, 1
  %v27 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsb6GslpvRJ9R_18cuda_oxide_coexist(i64 %v25, i64 %v26, i64 %v24) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v27, ptr %v15, align 8
  br label %bb6
bb4:
  %v28 = phi i64 [ %v59, %bb5 ], [ %v51, %bb6 ]
  %v29 = phi i64 [ %v60, %bb5 ], [ %v54, %bb6 ]
  %v30 = add i64 %v56, 1
  %v31 = icmp eq i64 %v30, 0
  %v32 = select i1 %v31, i8 0, i8 1
  %v33 = insertvalue { i8, { { i64 } } } undef, i8 %v32, 0
  %v34 = insertvalue { i8, { { i64 } } } %v33, i64 %v30, 1, 0, 0
  %v35 = extractvalue { i8, { { i64 } } } %v34, 0
  %v36 = zext i8 %v35 to i64
  %v37 = icmp eq i64 %v36, 1
  %v38 = extractvalue { i8, { { i64 } } } %v34, 1
  %v39 = alloca { { i64 } }, align 8
  store { { i64 } } %v38, ptr %v39, align 8
  %v40 = load i64, ptr %v39, align 8
  %v41 = icmp ugt i64 %v29, 0
  %v42 = xor i1 %v41, 1
  br i1 %v42, label %bb8, label %bb7
bb5:
  %v43 = extractvalue { ptr, i64 } %v10, 0
  %v44 = getelementptr inbounds float, ptr %v43, i64 %v63
  %v45 = load float, ptr %v44, align 4
  %v46 = extractvalue { ptr, i64 } %v11, 0
  %v47 = getelementptr inbounds float, ptr %v46, i64 %v63
  %v48 = fmul contract float %v45, %v12
  store float %v48, ptr %v47, align 4
  br label %bb4
bb6:
  %v49 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 0
  %v50 = getelementptr inbounds { i64, i64 }, ptr %v49, i32 0, i32 0
  %v51 = load i64, ptr %v50, align 8
  %v52 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 0
  %v53 = getelementptr inbounds { i64, i64 }, ptr %v52, i32 0, i32 1
  %v54 = load i64, ptr %v53, align 8
  %v55 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 1
  %v56 = load i64, ptr %v55, align 8
  %v57 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v15, i32 0, i32 2
  %v58 = load i1, ptr %v57, align 1
  br label %bb4
bb7:
  %v59 = add i64 %v28, %v40
  %v60 = sub i64 %v29, 1
  %v61 = insertvalue { i64, i64 } undef, i64 1, 0
  %v62 = insertvalue { i64, i64 } %v61, i64 %v28, 1
  %v63 = extractvalue { i64, i64 } %v62, 1
  %v64 = extractvalue { ptr, i64 } %v10, 1
  %v65 = icmp ult i64 %v63, %v64
  br i1 %v65, label %bb5, label %bb9
bb8:
  ret void
bb9:
  unreachable
}

define ptx_kernel void @write_pattern(ptr %v0, i64 %v1, i32 %v2) #0 {
entry:
  %v3 = insertvalue { ptr, i64 } undef, ptr %v0, 0
  %v4 = insertvalue { ptr, i64 } %v3, i64 %v1, 1
  br label %bb0
bb0:
  %v5 = phi { ptr, i64 } [ %v4, %entry ]
  %v6 = phi i32 [ %v2, %entry ]
  %v7 = alloca {  }, align 1
  %v8 = alloca { { i64, i64 }, i64, i1, [7 x i8] }, align 8
  %v9 = bitcast ptr %v7 to ptr
  %v10 = call i64 @cuda_device____internal__index_1d(ptr %v9) #0
  br label %bb1
bb1:
  %v11 = call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0
  br label %bb2
bb2:
  %v12 = call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0
  br label %bb3
bb3:
  %v13 = mul i32 %v11, %v12
  %v14 = zext i32 %v6 to i64
  %v15 = insertvalue { i64, i64 } undef, i64 %v10, 0
  %v16 = insertvalue { i64, i64 } %v15, i64 %v14, 1
  %v17 = zext i32 %v13 to i64
  %v18 = extractvalue { i64, i64 } %v16, 0
  %v19 = extractvalue { i64, i64 } %v16, 1
  %v20 = call { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsb6GslpvRJ9R_18cuda_oxide_coexist(i64 %v18, i64 %v19, i64 %v17) #0
  store { { i64, i64 }, i64, i1, [7 x i8] } %v20, ptr %v8, align 8
  br label %bb5
bb4:
  %v21 = phi i64 [ %v38, %bb5 ], [ %v46, %bb6 ]
  %v22 = phi i64 [ %v41, %bb5 ], [ %v47, %bb6 ]
  %v23 = add i64 %v43, 1
  %v24 = icmp eq i64 %v23, 0
  %v25 = select i1 %v24, i8 0, i8 1
  %v26 = insertvalue { i8, { { i64 } } } undef, i8 %v25, 0
  %v27 = insertvalue { i8, { { i64 } } } %v26, i64 %v23, 1, 0, 0
  %v28 = extractvalue { i8, { { i64 } } } %v27, 0
  %v29 = zext i8 %v28 to i64
  %v30 = icmp eq i64 %v29, 1
  %v31 = extractvalue { i8, { { i64 } } } %v27, 1
  %v32 = alloca { { i64 } }, align 8
  store { { i64 } } %v31, ptr %v32, align 8
  %v33 = load i64, ptr %v32, align 8
  %v34 = icmp ugt i64 %v22, 0
  %v35 = xor i1 %v34, 1
  br i1 %v35, label %bb7, label %bb6
bb5:
  %v36 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v8, i32 0, i32 0
  %v37 = getelementptr inbounds { i64, i64 }, ptr %v36, i32 0, i32 0
  %v38 = load i64, ptr %v37, align 8
  %v39 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v8, i32 0, i32 0
  %v40 = getelementptr inbounds { i64, i64 }, ptr %v39, i32 0, i32 1
  %v41 = load i64, ptr %v40, align 8
  %v42 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v8, i32 0, i32 1
  %v43 = load i64, ptr %v42, align 8
  %v44 = getelementptr inbounds { { i64, i64 }, i64, i1, [7 x i8] }, ptr %v8, i32 0, i32 2
  %v45 = load i1, ptr %v44, align 1
  br label %bb4
bb6:
  %v46 = add i64 %v21, %v33
  %v47 = sub i64 %v22, 1
  %v48 = insertvalue { i64, i64 } undef, i64 1, 0
  %v49 = insertvalue { i64, i64 } %v48, i64 %v21, 1
  %v50 = extractvalue { i64, i64 } %v49, 1
  %v51 = uitofp i64 %v50 to float
  %v52 = extractvalue { ptr, i64 } %v5, 0
  %v53 = getelementptr inbounds float, ptr %v52, i64 %v50
  %v54 = fadd contract float 42.0, %v51
  store float %v54, ptr %v53, align 4
  br label %bb4
bb7:
  ret void
}

declare i32 @llvm.nvvm.read.ptx.sreg.tid.x()
declare i32 @llvm.nvvm.read.ptx.sreg.ctaid.x()

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

define { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsb6GslpvRJ9R_18cuda_oxide_coexist(i64 %v0, i64 %v1, i64 %v2) #0 {
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


attributes #0 = { convergent }
