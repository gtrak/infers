; ModuleID = '/home/gary/dev/infers/crates/cuda-oxide-coexist/cuda_oxide_coexist.ll'
source_filename = "cuda_oxide_coexist"
target datalayout = "e-i64:64-i128:128-v16:16-v32:32-n16:32:64"
target triple = "nvptx64-nvidia-cuda"

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, 1025) i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 1, -2147483648) i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #0

; Function Attrs: convergent nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @mem_copy(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(none) %v2, i64 %v3, i32 %v4) local_unnamed_addr #1 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7
  %v3.i = zext nneg i32 %v2.i to i64
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7
  %v5.i = zext nneg i32 %v4.i to i64
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7
  %v7.i = zext nneg i32 %v6.i to i64
  %v8.i = mul nuw nsw i64 %v5.i, %v7.i
  %v9.i = add nuw nsw i64 %v8.i, %v3.i
  %v17 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #7
  %v18 = mul i32 %v6.i, %v17
  %v19 = zext i32 %v4 to i64
  %v22 = zext i32 %v18 to i64
  %spec.select.i.i = tail call i64 @llvm.usub.sat.i64(i64 %v19, i64 %v9.i)
  %v25.i.i.lhs.trunc = trunc nuw i64 %spec.select.i.i to i32
  %v25.i.i.lhs.trunc.frozen = freeze i32 %v25.i.i.lhs.trunc
  %v25.i.i1 = udiv i32 %v25.i.i.lhs.trunc.frozen, %v18
  %v25.i.i.zext = zext i32 %v25.i.i1 to i64
  %0 = mul i32 %v25.i.i1, %v18
  %v26.i.i2.decomposed = sub i32 %v25.i.i.lhs.trunc.frozen, %0
  %v27.not.i.i = icmp ne i32 %v26.i.i2.decomposed, 0
  %v29.i.i = zext i1 %v27.not.i.i to i64
  %v30.i.i = add nuw nsw i64 %v29.i.i, %v25.i.i.zext
  %v39.not3 = icmp eq i64 %v30.i.i, 0
  br i1 %v39.not3, label %bb8, label %bb7.preheader

bb7.preheader:                                    ; preds = %entry
  %1 = add nuw nsw i64 %v25.i.i.zext, %v29.i.i
  %2 = add nsw i64 %1, -1
  %xtraiter = and i64 %v30.i.i, 3
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb7.prol.loopexit, label %bb7.prol

bb7.prol:                                         ; preds = %bb7.preheader, %bb7.prol
  %v275.prol = phi i64 [ %v57.prol, %bb7.prol ], [ %v30.i.i, %bb7.preheader ]
  %v264.prol = phi i64 [ %v56.prol, %bb7.prol ], [ %v9.i, %bb7.preheader ]
  %prol.iter = phi i64 [ %prol.iter.next, %bb7.prol ], [ 0, %bb7.preheader ]
  %v56.prol = add i64 %v264.prol, %v22
  %v57.prol = add nsw i64 %v275.prol, -1
  %v62.prol = icmp ult i64 %v264.prol, %v1
  tail call void @llvm.assume(i1 %v62.prol)
  %v42.prol = getelementptr inbounds float, ptr %v0, i64 %v264.prol
  %v43.prol = load float, ptr %v42.prol, align 4
  %v45.prol = getelementptr inbounds float, ptr %v2, i64 %v264.prol
  store float %v43.prol, ptr %v45.prol, align 4
  %prol.iter.next = add i64 %prol.iter, 1
  %prol.iter.cmp.not = icmp eq i64 %prol.iter.next, %xtraiter
  br i1 %prol.iter.cmp.not, label %bb7.prol.loopexit, label %bb7.prol, !llvm.loop !0

bb7.prol.loopexit:                                ; preds = %bb7.prol, %bb7.preheader
  %v275.unr = phi i64 [ %v30.i.i, %bb7.preheader ], [ %v57.prol, %bb7.prol ]
  %v264.unr = phi i64 [ %v9.i, %bb7.preheader ], [ %v56.prol, %bb7.prol ]
  %3 = icmp ult i64 %2, 3
  br i1 %3, label %bb8, label %bb7

bb7:                                              ; preds = %bb7.prol.loopexit, %bb7
  %v275 = phi i64 [ %v57.3, %bb7 ], [ %v275.unr, %bb7.prol.loopexit ]
  %v264 = phi i64 [ %v56.3, %bb7 ], [ %v264.unr, %bb7.prol.loopexit ]
  %v56 = add i64 %v264, %v22
  %v62 = icmp ult i64 %v264, %v1
  tail call void @llvm.assume(i1 %v62)
  %v42 = getelementptr inbounds float, ptr %v0, i64 %v264
  %v43 = load float, ptr %v42, align 4
  %v45 = getelementptr inbounds float, ptr %v2, i64 %v264
  store float %v43, ptr %v45, align 4
  %v56.1 = add i64 %v56, %v22
  %v62.1 = icmp ult i64 %v56, %v1
  tail call void @llvm.assume(i1 %v62.1)
  %v42.1 = getelementptr inbounds float, ptr %v0, i64 %v56
  %v43.1 = load float, ptr %v42.1, align 4
  %v45.1 = getelementptr inbounds float, ptr %v2, i64 %v56
  store float %v43.1, ptr %v45.1, align 4
  %v56.2 = add i64 %v56.1, %v22
  %v62.2 = icmp ult i64 %v56.1, %v1
  tail call void @llvm.assume(i1 %v62.2)
  %v42.2 = getelementptr inbounds float, ptr %v0, i64 %v56.1
  %v43.2 = load float, ptr %v42.2, align 4
  %v45.2 = getelementptr inbounds float, ptr %v2, i64 %v56.1
  store float %v43.2, ptr %v45.2, align 4
  %v56.3 = add i64 %v56.2, %v22
  %v57.3 = add nsw i64 %v275, -4
  %v62.3 = icmp ult i64 %v56.2, %v1
  tail call void @llvm.assume(i1 %v62.3)
  %v42.3 = getelementptr inbounds float, ptr %v0, i64 %v56.2
  %v43.3 = load float, ptr %v42.3, align 4
  %v45.3 = getelementptr inbounds float, ptr %v2, i64 %v56.2
  store float %v43.3, ptr %v45.3, align 4
  %v39.not.3 = icmp eq i64 %v57.3, 0
  br i1 %v39.not.3, label %bb8, label %bb7

bb8:                                              ; preds = %bb7.prol.loopexit, %bb7, %entry
  ret void
}

; Function Attrs: convergent nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: write)
define ptx_kernel void @scalar_mul(ptr readonly captures(none) %v0, i64 %v1, ptr writeonly captures(none) %v2, i64 %v3, float %v4, i32 %v5) local_unnamed_addr #1 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7
  %v3.i = zext nneg i32 %v2.i to i64
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7
  %v5.i = zext nneg i32 %v4.i to i64
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7
  %v7.i = zext nneg i32 %v6.i to i64
  %v8.i = mul nuw nsw i64 %v5.i, %v7.i
  %v9.i = add nuw nsw i64 %v8.i, %v3.i
  %v19 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #7
  %v20 = mul i32 %v6.i, %v19
  %v21 = zext i32 %v5 to i64
  %v24 = zext i32 %v20 to i64
  %spec.select.i.i = tail call i64 @llvm.usub.sat.i64(i64 %v21, i64 %v9.i)
  %v25.i.i.lhs.trunc = trunc nuw i64 %spec.select.i.i to i32
  %v25.i.i.lhs.trunc.frozen = freeze i32 %v25.i.i.lhs.trunc
  %v25.i.i1 = udiv i32 %v25.i.i.lhs.trunc.frozen, %v20
  %v25.i.i.zext = zext i32 %v25.i.i1 to i64
  %0 = mul i32 %v25.i.i1, %v20
  %v26.i.i2.decomposed = sub i32 %v25.i.i.lhs.trunc.frozen, %0
  %v27.not.i.i = icmp ne i32 %v26.i.i2.decomposed, 0
  %v29.i.i = zext i1 %v27.not.i.i to i64
  %v30.i.i = add nuw nsw i64 %v29.i.i, %v25.i.i.zext
  %v41.not3 = icmp eq i64 %v30.i.i, 0
  br i1 %v41.not3, label %bb8, label %bb7.preheader

bb7.preheader:                                    ; preds = %entry
  %1 = add nuw nsw i64 %v25.i.i.zext, %v29.i.i
  %xtraiter = and i64 %v30.i.i, 1
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb7.prol.loopexit, label %bb7.prol

bb7.prol:                                         ; preds = %bb7.preheader
  %v59.prol = add nuw nsw i64 %v9.i, %v24
  %v60.prol = add nsw i64 %v30.i.i, -1
  %v65.prol = icmp ult i64 %v9.i, %v1
  tail call void @llvm.assume(i1 %v65.prol)
  %v44.prol = getelementptr inbounds nuw float, ptr %v0, i64 %v9.i
  %v45.prol = load float, ptr %v44.prol, align 4
  %v47.prol = getelementptr inbounds nuw float, ptr %v2, i64 %v9.i
  %v48.prol = fmul contract float %v4, %v45.prol
  store float %v48.prol, ptr %v47.prol, align 4
  br label %bb7.prol.loopexit

bb7.prol.loopexit:                                ; preds = %bb7.prol, %bb7.preheader
  %v295.unr = phi i64 [ %v30.i.i, %bb7.preheader ], [ %v60.prol, %bb7.prol ]
  %v284.unr = phi i64 [ %v9.i, %bb7.preheader ], [ %v59.prol, %bb7.prol ]
  %2 = icmp eq i64 %1, 1
  br i1 %2, label %bb8, label %bb7

bb7:                                              ; preds = %bb7.prol.loopexit, %bb7
  %v295 = phi i64 [ %v60.1, %bb7 ], [ %v295.unr, %bb7.prol.loopexit ]
  %v284 = phi i64 [ %v59.1, %bb7 ], [ %v284.unr, %bb7.prol.loopexit ]
  %v59 = add i64 %v284, %v24
  %v65 = icmp ult i64 %v284, %v1
  tail call void @llvm.assume(i1 %v65)
  %v44 = getelementptr inbounds float, ptr %v0, i64 %v284
  %v45 = load float, ptr %v44, align 4
  %v47 = getelementptr inbounds float, ptr %v2, i64 %v284
  %v48 = fmul contract float %v4, %v45
  store float %v48, ptr %v47, align 4
  %v59.1 = add i64 %v59, %v24
  %v60.1 = add nsw i64 %v295, -2
  %v65.1 = icmp ult i64 %v59, %v1
  tail call void @llvm.assume(i1 %v65.1)
  %v44.1 = getelementptr inbounds float, ptr %v0, i64 %v59
  %v45.1 = load float, ptr %v44.1, align 4
  %v47.1 = getelementptr inbounds float, ptr %v2, i64 %v59
  %v48.1 = fmul contract float %v4, %v45.1
  store float %v48.1, ptr %v47.1, align 4
  %v41.not.1 = icmp eq i64 %v60.1, 0
  br i1 %v41.not.1, label %bb8, label %bb7

bb8:                                              ; preds = %bb7.prol.loopexit, %bb7, %entry
  ret void
}

; Function Attrs: convergent nofree norecurse nosync nounwind memory(argmem: write)
define ptx_kernel void @write_pattern(ptr writeonly captures(none) %v0, i64 %v1, i32 %v2) local_unnamed_addr #2 {
entry:
  %v2.i = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7
  %v3.i = zext nneg i32 %v2.i to i64
  %v4.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7
  %v5.i = zext nneg i32 %v4.i to i64
  %v6.i = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7
  %v7.i = zext nneg i32 %v6.i to i64
  %v8.i = mul nuw nsw i64 %v5.i, %v7.i
  %v9.i = add nuw nsw i64 %v8.i, %v3.i
  %v12 = tail call i32 @llvm.nvvm.read.ptx.sreg.nctaid.x() #7
  %v13 = mul i32 %v6.i, %v12
  %v14 = zext i32 %v2 to i64
  %v17 = zext i32 %v13 to i64
  %spec.select.i.i = tail call i64 @llvm.usub.sat.i64(i64 %v14, i64 %v9.i)
  %v25.i.i.lhs.trunc = trunc nuw i64 %spec.select.i.i to i32
  %v25.i.i.lhs.trunc.frozen = freeze i32 %v25.i.i.lhs.trunc
  %v25.i.i1 = udiv i32 %v25.i.i.lhs.trunc.frozen, %v13
  %v25.i.i.zext = zext i32 %v25.i.i1 to i64
  %0 = mul i32 %v25.i.i1, %v13
  %v26.i.i2.decomposed = sub i32 %v25.i.i.lhs.trunc.frozen, %0
  %v27.not.i.i = icmp ne i32 %v26.i.i2.decomposed, 0
  %v29.i.i = zext i1 %v27.not.i.i to i64
  %v30.i.i = add nuw nsw i64 %v29.i.i, %v25.i.i.zext
  %v34.not3 = icmp eq i64 %v30.i.i, 0
  br i1 %v34.not3, label %bb7, label %bb6.preheader

bb6.preheader:                                    ; preds = %entry
  %1 = add nuw nsw i64 %v25.i.i.zext, %v29.i.i
  %2 = add nsw i64 %1, -1
  %xtraiter = and i64 %v30.i.i, 3
  %lcmp.mod.not = icmp eq i64 %xtraiter, 0
  br i1 %lcmp.mod.not, label %bb6.prol.loopexit, label %bb6.prol

bb6.prol:                                         ; preds = %bb6.preheader, %bb6.prol
  %v225.prol = phi i64 [ %v47.prol, %bb6.prol ], [ %v30.i.i, %bb6.preheader ]
  %v214.prol = phi i64 [ %v46.prol, %bb6.prol ], [ %v9.i, %bb6.preheader ]
  %prol.iter = phi i64 [ %prol.iter.next, %bb6.prol ], [ 0, %bb6.preheader ]
  %v46.prol = add i64 %v214.prol, %v17
  %v47.prol = add nsw i64 %v225.prol, -1
  %v51.prol = uitofp i64 %v214.prol to float
  %v53.prol = getelementptr inbounds float, ptr %v0, i64 %v214.prol
  %v54.prol = fadd contract float %v51.prol, 4.200000e+01
  store float %v54.prol, ptr %v53.prol, align 4
  %prol.iter.next = add i64 %prol.iter, 1
  %prol.iter.cmp.not = icmp eq i64 %prol.iter.next, %xtraiter
  br i1 %prol.iter.cmp.not, label %bb6.prol.loopexit, label %bb6.prol, !llvm.loop !2

bb6.prol.loopexit:                                ; preds = %bb6.prol, %bb6.preheader
  %v225.unr = phi i64 [ %v30.i.i, %bb6.preheader ], [ %v47.prol, %bb6.prol ]
  %v214.unr = phi i64 [ %v9.i, %bb6.preheader ], [ %v46.prol, %bb6.prol ]
  %3 = icmp ult i64 %2, 3
  br i1 %3, label %bb7, label %bb6

bb6:                                              ; preds = %bb6.prol.loopexit, %bb6
  %v225 = phi i64 [ %v47.3, %bb6 ], [ %v225.unr, %bb6.prol.loopexit ]
  %v214 = phi i64 [ %v46.3, %bb6 ], [ %v214.unr, %bb6.prol.loopexit ]
  %v46 = add i64 %v214, %v17
  %v51 = uitofp i64 %v214 to float
  %v53 = getelementptr inbounds float, ptr %v0, i64 %v214
  %v54 = fadd contract float %v51, 4.200000e+01
  store float %v54, ptr %v53, align 4
  %v46.1 = add i64 %v46, %v17
  %v51.1 = uitofp i64 %v46 to float
  %v53.1 = getelementptr inbounds float, ptr %v0, i64 %v46
  %v54.1 = fadd contract float %v51.1, 4.200000e+01
  store float %v54.1, ptr %v53.1, align 4
  %v46.2 = add i64 %v46.1, %v17
  %v51.2 = uitofp i64 %v46.1 to float
  %v53.2 = getelementptr inbounds float, ptr %v0, i64 %v46.1
  %v54.2 = fadd contract float %v51.2, 4.200000e+01
  store float %v54.2, ptr %v53.2, align 4
  %v46.3 = add i64 %v46.2, %v17
  %v47.3 = add nsw i64 %v225, -4
  %v51.3 = uitofp i64 %v46.2 to float
  %v53.3 = getelementptr inbounds float, ptr %v0, i64 %v46.2
  %v54.3 = fadd contract float %v51.3, 4.200000e+01
  store float %v54.3, ptr %v53.3, align 4
  %v34.not.3 = icmp eq i64 %v47.3, 0
  br i1 %v34.not.3, label %bb7, label %bb6

bb7:                                              ; preds = %bb6.prol.loopexit, %bb6, %entry
  ret void
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 1024) i32 @llvm.nvvm.read.ptx.sreg.tid.x() #0

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare noundef range(i32 0, 2147483647) i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #0

; Function Attrs: alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define range(i64 0, 2199023254528) i64 @cuda_device____internal__index_1d(ptr readnone captures(none) %v0) local_unnamed_addr #3 {
entry:
  %v2 = tail call i32 @llvm.nvvm.read.ptx.sreg.tid.x() #7
  %v3 = zext nneg i32 %v2 to i64
  %v4 = tail call i32 @llvm.nvvm.read.ptx.sreg.ctaid.x() #7
  %v5 = zext nneg i32 %v4 to i64
  %v6 = tail call i32 @llvm.nvvm.read.ptx.sreg.ntid.x() #7
  %v7 = zext nneg i32 %v6 to i64
  %v8 = mul nuw nsw i64 %v5, %v7
  %v9 = add nuw nsw i64 %v8, %v3
  ret i64 %v9
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define { { i64, i64 }, i64, i1, [7 x i8] } @_RNvMNtNtNtCsiQ4CSjCKWVc_4core4iter8adapters7step_byINtB2_6StepByINtNtNtB8_3ops5range5RangejEE3newCsb6GslpvRJ9R_18cuda_oxide_coexist(i64 %v0, i64 %v1, i64 %v2) local_unnamed_addr #4 {
entry:
  %spec.select.i = tail call i64 @llvm.usub.sat.i64(i64 %v1, i64 %v0)
  %spec.select.i.frozen = freeze i64 %spec.select.i
  %v2.frozen = freeze i64 %v2
  %v25.i = udiv i64 %spec.select.i.frozen, %v2.frozen
  %0 = mul i64 %v25.i, %v2.frozen
  %v26.i.decomposed = sub i64 %spec.select.i.frozen, %0
  %v27.not.i = icmp ne i64 %v26.i.decomposed, 0
  %v29.i = zext i1 %v27.not.i to i64
  %v30.i = add i64 %v25.i, %v29.i
  %v32.fca.0.insert.i = insertvalue { i64, i64 } poison, i64 %v0, 0
  %v32.fca.1.insert.i = insertvalue { i64, i64 } %v32.fca.0.insert.i, i64 %v30.i, 1
  %v11 = add i64 %v2, -1
  %v12 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } undef, { i64, i64 } %v32.fca.1.insert.i, 0
  %v13 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } %v12, i64 %v11, 1
  %v14 = insertvalue { { i64, i64 }, i64, i1, [7 x i8] } %v13, i1 true, 2
  ret { { i64, i64 }, i64, i1, [7 x i8] } %v14
}

; Function Attrs: mustprogress nofree norecurse nosync nounwind willreturn memory(none)
define { i64, i64 } @_std__ops__Range_usize__as_std__iter__adapters__step_by__SpecRangeSetup_std__ops__Range_usize_____setup(i64 %v0, i64 %v1, i64 %v2) local_unnamed_addr #4 {
entry:
  %spec.select = tail call i64 @llvm.usub.sat.i64(i64 %v1, i64 %v0)
  %spec.select.frozen = freeze i64 %spec.select
  %v2.frozen = freeze i64 %v2
  %v25 = udiv i64 %spec.select.frozen, %v2.frozen
  %0 = mul i64 %v25, %v2.frozen
  %v26.decomposed = sub i64 %spec.select.frozen, %0
  %v27.not = icmp ne i64 %v26.decomposed, 0
  %v29 = zext i1 %v27.not to i64
  %v30 = add i64 %v25, %v29
  %v32.fca.0.insert = insertvalue { i64, i64 } poison, i64 %v0, 0
  %v32.fca.1.insert = insertvalue { i64, i64 } %v32.fca.0.insert, i64 %v30, 1
  ret { i64, i64 } %v32.fca.1.insert
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write)
declare void @llvm.assume(i1 noundef) #5

; Function Attrs: nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none)
declare i64 @llvm.usub.sat.i64(i64, i64) #6

attributes #0 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #1 = { convergent nofree norecurse nosync nounwind memory(argmem: readwrite, inaccessiblemem: write) }
attributes #2 = { convergent nofree norecurse nosync nounwind memory(argmem: write) }
attributes #3 = { alwaysinline convergent mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #4 = { mustprogress nofree norecurse nosync nounwind willreturn memory(none) }
attributes #5 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: write) }
attributes #6 = { nocallback nocreateundeforpoison nofree nosync nounwind speculatable willreturn memory(none) }
attributes #7 = { convergent }

!0 = distinct !{!0, !1}
!1 = !{!"llvm.loop.unroll.disable"}
!2 = distinct !{!2, !1}
