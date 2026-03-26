#include "pto/pto-inst.hpp"
using namespace pto;

enum class PTOAutoSyncTailMode : int {
  kBarrierAll = 0,
  kSetWaitMte3ToSEvent0 = 1,
};

static AICORE inline void ptoas_auto_sync_tail(
    PTOAutoSyncTailMode mode = PTOAutoSyncTailMode::kBarrierAll) {
  switch (mode) {
  case PTOAutoSyncTailMode::kSetWaitMte3ToSEvent0:
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
    break;
  case PTOAutoSyncTailMode::kBarrierAll:
  default:
    pipe_barrier(PIPE_ALL);
    break;
  }
}

__global__ AICORE void fa_perf_kernel(__gm__ half* v1, __gm__ half* v2, __gm__ half* v3, __gm__ half* v4, __gm__ float* v5, __gm__ half* v6, __gm__ float* v7, int32_t v8, int32_t v9, int32_t v10, int32_t v11, __gm__ int64_t* v12) {
  RoundMode v13 = RoundMode::CAST_ROUND;
  unsigned v14 = 64;
  unsigned v15 = 128;
  unsigned v16 = 1;
  unsigned v17 = 0;
  int32_t v18 = 1;
  int32_t v19 = 24576;
  int32_t v20 = 128;
  int32_t v21 = 0;
  int32_t v22 = 2;
  int64_t v23 = 0;
  int64_t v24 = 32768;
  int64_t v25 = 65536;
  int64_t v26 = 98304;
  int32_t v27 = 3;
  int32_t v28 = 4;
  int32_t v29 = 5;
  int32_t v30 = 6;
  int32_t v31 = 7;
  int32_t v32 = 8;
  int32_t v33 = 9;
  int32_t v34 = 10;
  int32_t v35 = 11;
  int32_t v36 = 12;
  int32_t v37 = 13;
  int32_t v38 = 14;
  int32_t v39 = 15;
  int64_t v40 = 196608;
  int64_t v41 = 229376;
  int64_t v42 = 131072;
  int64_t v43 = 163840;
  int32_t v44 = 512;
  int32_t v45 = 64;
  int64_t v46 = 82176;
  int64_t v47 = 82432;
  int64_t v48 = 82688;
  int64_t v49 = 82944;
  float v50 = 1.0f;
  float v51 = 0.0883883535f;
  int64_t v52 = 83200;
  int64_t v53 = 83456;
  int64_t v54 = 81920;
  int64_t v55 = 83712;
  int64_t v56 = 116480;
  int64_t v57 = 149248;
  int32_t v58 = 127;
  using T = float;
  int64_t v59 = (int64_t) ((int64_t) v53);
  int64_t v60 = (int64_t) ((int64_t) v52);
  int32_t v61 = (int32_t) ((size_t) v29);
  int32_t v62 = (int32_t) ((size_t) v27);
  int64_t v63 = (int64_t) ((int64_t) v26);
  int64_t v64 = (int64_t) ((int64_t) v25);
  int64_t v65 = (int64_t) ((int64_t) v24);
  int64_t v66 = (int64_t) ((int64_t) v23);
  size_t v67 = (size_t) v21;
  int32_t v68 = (int32_t) v67;
  size_t v69 = (size_t) v18;
  int32_t v70 = (int32_t) v69;
  uint64_t v71 = (uint64_t) v12;
  set_ffts_base_addr(v71);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v72;
  TASSIGN(v72, v23);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v73;
  TASSIGN(v73, v24);
  Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v74;
  TASSIGN(v74, v25);
  Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, 64, 1, SLayout::NoneBox, 512, PadValue::Null> v75;
  TASSIGN(v75, v54);
  Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v76;
  TASSIGN(v76, v54);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v77;
  TASSIGN(v77, v55);
  Tile<TileType::Vec, float, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v78;
  TASSIGN(v78, v56);
  Tile<TileType::Vec, half, 64, 128, BLayout::RowMajor, 64, 128, SLayout::NoneBox, 512, PadValue::Null> v79;
  TASSIGN(v79, v57);
  int32_t v80 = (int32_t) ((size_t) (int32_t) ((uint32_t) v8 + (uint32_t) v58) / v20);
  int32_t v81 = (int32_t) ((uint32_t) v10 + (uint32_t) v58) / v20;
  size_t v82 = (size_t) v81;
  int64_t v83 = get_block_num();
  int32_t v84 = (int32_t) ((size_t) (int32_t) ((int64_t) v83));
  int64_t v85 = get_block_idx();
  int32_t v86 = (int32_t) ((int64_t) v85);
  int32_t v87 = (int32_t) ((size_t) v86);

  #if defined(__DAV_CUBE__)
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  set_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  int32_t v88;
  int32_t v89;
  int32_t v90;
  int32_t v91;
  int32_t v92;
  int32_t v93;
  v88 = v68;
  v89 = v68;
  v90 = v68;
  v91 = v68;
  v92 = v68;
  v93 = v68;
  for (int32_t v94 = v87; v94 < v80; v94 += v84) {
    int32_t v95 = (int32_t) ((uint32_t) v94 * (uint32_t) v20);
    int32_t v96;
    int32_t v97;
    int32_t v98;
    int32_t v99;
    v96 = v88;
    v97 = v89;
    v98 = v90;
    v99 = v93;
    for (int32_t v100 = v68; v100 < v70; v100 += v70) {
      uint32_t v101 = (uint32_t) v91;
      int32_t v102 = (int32_t) ((uint32_t) ((int32_t) v101 * (uint32_t) v81) + (uint32_t) v100) % v22;
      int32_t v103 = v91 % v22;
      int32_t v104 = v100 % v22;
      int32_t v105 = (int32_t) ((uint32_t) v100 * (uint32_t) v20);
      bool v106 = v102 == v21;
      int32_t v107;
      if (v106) {
        v107 = v70;
      } else {
        v107 = v70;
      };
      event_t v108 = (event_t) v107;
      wait_flag(PIPE_MTE1, PIPE_MTE2, v108);
      if (v100 == v21) {
        int64_t v109;
        if (v103 == v21) {
          v109 = v66;
        } else {
          v109 = v65;
        };
        Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v110;
        TASSIGN(v110, v109);
        unsigned v111 = (unsigned) v9;
        unsigned v112 = v15 * v111;
        pto::Shape<1, 1, 1, 128, 128> v113 = pto::Shape<1, 1, 1, 128, 128>();
        pto::Stride<-1, -1, -1, -1, 1> v114 = pto::Stride<-1, -1, -1, -1, 1>(v112, v112, v112, v111);
        GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v115 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v17 + (unsigned) v95 * (unsigned) v9 + v17 * (unsigned) v18), v113, v114);
        TLOAD(v110, v115);
      };
      int64_t v116;
      if (v106) {
        v116 = v64;
      } else {
        v116 = v63;
      };
      Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v117;
      TASSIGN(v117, v116);
      pto::Shape<1, 1, 1, 128, 128> v118 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<128, 128, 128, 1, -1> v119 = pto::Stride<128, 128, 128, 1, -1>((unsigned) v9);
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<128, 128, 128, 1, -1>, pto::Layout::DN> v120 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<128, 128, 128, 1, -1>, pto::Layout::DN>(v2 + (v17 + v17 * (unsigned) v18 + (unsigned) v105 * (unsigned) v9), v118, v119);
      TLOAD(v117, v120);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      bool v121 = v97 == v21;
      int32_t v122;
      if (v121) {
        v122 = v70;
      } else {
        v122 = v70;
      };
      event_t v123 = (event_t) v122;
      wait_flag(PIPE_M, PIPE_MTE1, v123);
      int64_t v124;
      if (v121) {
        v124 = v66;
      } else {
        v124 = v65;
      };
      Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v125;
      TASSIGN(v125, v124);
      int64_t v126;
      if (v103 == v21) {
        v126 = v66;
      } else {
        v126 = v65;
      };
      Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v127;
      TASSIGN(v127, v126);
      TMOV(v125, v127);
      Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v128;
      TASSIGN(v128, v124);
      Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v129;
      TASSIGN(v129, v116);
      TMOV(v128, v129);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      event_t v130 = (event_t) v107;
      set_flag(PIPE_MTE1, PIPE_MTE2, v130);
      bool v131 = v98 == v21;
      int32_t v132;
      if (v131) {
        v132 = v70;
      } else {
        v132 = v70;
      };
      event_t v133 = (event_t) v132;
      wait_flag(PIPE_FIX, PIPE_M, v133);
      int64_t v134;
      if (v131) {
        v134 = v66;
      } else {
        v134 = v64;
      };
      Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v135;
      TASSIGN(v135, v134);
      Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v136;
      TASSIGN(v136, v124);
      Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v137;
      TASSIGN(v137, v124);
      TMATMUL(v135, v136, v137);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      event_t v138 = (event_t) v122;
      set_flag(PIPE_M, PIPE_MTE1, v138);
      Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v139;
      TASSIGN(v139, v134);
      unsigned v140 = (unsigned) v10;
      unsigned v141 = v15 * v140;
      pto::Shape<1, 1, 1, 128, 128> v142 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v143 = pto::Stride<-1, -1, -1, -1, 1>(v141, v141, v141, v140);
      GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v144 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v5 + (v17 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v104 * (uint32_t) v8) + (uint32_t) v95) * (unsigned) v10 + (unsigned) v105 * (unsigned) v18), v142, v143);
      TSTORE(v144, v139);
      event_t v145 = (event_t) v132;
      set_flag(PIPE_FIX, PIPE_M, v145);
      uint32_t v146 = (uint32_t) v97;
      uint32_t v147 = (uint32_t) v98;
      int32_t v148;
      if (v104 == v21) {
        v148 = v70;
      } else {
        v148 = v70;
      };
      bool v149 = v148 == v21;
      if (v149) {
        uint16_t v150 = getFFTSMsg(FFTS_MODE_VAL, v21);
        ffts_cross_core_sync(PIPE_FIX, v150);
      };
      bool v151 = v148 == v18;
      if (v151) {
        uint16_t v152 = getFFTSMsg(FFTS_MODE_VAL, v18);
        ffts_cross_core_sync(PIPE_FIX, v152);
      };
      bool v153 = v148 == v22;
      if (v153) {
        uint16_t v154 = getFFTSMsg(FFTS_MODE_VAL, v22);
        ffts_cross_core_sync(PIPE_FIX, v154);
      };
      bool v155 = v148 == v27;
      if (v155) {
        uint16_t v156 = getFFTSMsg(FFTS_MODE_VAL, v27);
        ffts_cross_core_sync(PIPE_FIX, v156);
      };
      bool v157 = v148 == v28;
      if (v157) {
        uint16_t v158 = getFFTSMsg(FFTS_MODE_VAL, v28);
        ffts_cross_core_sync(PIPE_FIX, v158);
      };
      bool v159 = v148 == v29;
      if (v159) {
        uint16_t v160 = getFFTSMsg(FFTS_MODE_VAL, v29);
        ffts_cross_core_sync(PIPE_FIX, v160);
      };
      bool v161 = v148 == v30;
      if (v161) {
        uint16_t v162 = getFFTSMsg(FFTS_MODE_VAL, v30);
        ffts_cross_core_sync(PIPE_FIX, v162);
      };
      bool v163 = v148 == v31;
      if (v163) {
        uint16_t v164 = getFFTSMsg(FFTS_MODE_VAL, v31);
        ffts_cross_core_sync(PIPE_FIX, v164);
      };
      bool v165 = v148 == v32;
      if (v165) {
        uint16_t v166 = getFFTSMsg(FFTS_MODE_VAL, v32);
        ffts_cross_core_sync(PIPE_FIX, v166);
      };
      bool v167 = v148 == v33;
      if (v167) {
        uint16_t v168 = getFFTSMsg(FFTS_MODE_VAL, v33);
        ffts_cross_core_sync(PIPE_FIX, v168);
      };
      bool v169 = v148 == v34;
      if (v169) {
        uint16_t v170 = getFFTSMsg(FFTS_MODE_VAL, v34);
        ffts_cross_core_sync(PIPE_FIX, v170);
      };
      bool v171 = v148 == v35;
      if (v171) {
        uint16_t v172 = getFFTSMsg(FFTS_MODE_VAL, v35);
        ffts_cross_core_sync(PIPE_FIX, v172);
      };
      bool v173 = v148 == v36;
      if (v173) {
        uint16_t v174 = getFFTSMsg(FFTS_MODE_VAL, v36);
        ffts_cross_core_sync(PIPE_FIX, v174);
      };
      bool v175 = v148 == v37;
      if (v175) {
        uint16_t v176 = getFFTSMsg(FFTS_MODE_VAL, v37);
        ffts_cross_core_sync(PIPE_FIX, v176);
      };
      bool v177 = v148 == v38;
      if (v177) {
        uint16_t v178 = getFFTSMsg(FFTS_MODE_VAL, v38);
        ffts_cross_core_sync(PIPE_FIX, v178);
      };
      bool v179 = v148 == v39;
      if (v179) {
        uint16_t v180 = getFFTSMsg(FFTS_MODE_VAL, v39);
        ffts_cross_core_sync(PIPE_FIX, v180);
      };
      v96 = (int32_t) ((size_t) v102);
      v97 = (int32_t) ((size_t) (int32_t) ((uint32_t) v18 - v146));
      v98 = (int32_t) ((size_t) (int32_t) ((uint32_t) v18 - v147));
      v99 = v100;
    };
    int32_t v181;
    int32_t v182;
    int32_t v183;
    int32_t v184;
    v181 = v96;
    v182 = v97;
    v183 = v98;
    v184 = v99;
    for (int32_t v185 = v68; v185 < ((int32_t) v82); v185 += v70) {
      int32_t v186 = (int32_t) ((uint32_t) v185 + (uint32_t) v18);
      int32_t v187;
      int32_t v188;
      int32_t v189;
      int32_t v190;
      if (v186 < v81) {
        uint32_t v191 = (uint32_t) v91;
        int32_t v192 = (int32_t) ((uint32_t) ((int32_t) v191 * (uint32_t) v81) + (uint32_t) v186) % v22;
        int32_t v193 = v91 % v22;
        int32_t v194 = v186 % v22;
        int32_t v195 = (int32_t) ((uint32_t) v186 * (uint32_t) v20);
        bool v196 = v192 == v21;
        int32_t v197;
        if (v196) {
          v197 = v70;
        } else {
          v197 = v70;
        };
        event_t v198 = (event_t) v197;
        wait_flag(PIPE_MTE1, PIPE_MTE2, v198);
        if (v186 == v21) {
          int64_t v199;
          if (v193 == v21) {
            v199 = v66;
          } else {
            v199 = v65;
          };
          Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v200;
          TASSIGN(v200, v199);
          unsigned v201 = (unsigned) v9;
          unsigned v202 = v15 * v201;
          pto::Shape<1, 1, 1, 128, 128> v203 = pto::Shape<1, 1, 1, 128, 128>();
          pto::Stride<-1, -1, -1, -1, 1> v204 = pto::Stride<-1, -1, -1, -1, 1>(v202, v202, v202, v201);
          GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v205 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v1 + (v17 + (unsigned) v95 * (unsigned) v9 + v17 * (unsigned) v18), v203, v204);
          TLOAD(v200, v205);
        };
        int64_t v206;
        if (v196) {
          v206 = v64;
        } else {
          v206 = v63;
        };
        Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v207;
        TASSIGN(v207, v206);
        pto::Shape<1, 1, 1, 128, 128> v208 = pto::Shape<1, 1, 1, 128, 128>();
        pto::Stride<128, 128, 128, 1, -1> v209 = pto::Stride<128, 128, 128, 1, -1>((unsigned) v9);
        GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<128, 128, 128, 1, -1>, pto::Layout::DN> v210 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<128, 128, 128, 1, -1>, pto::Layout::DN>(v2 + (v17 + v17 * (unsigned) v18 + (unsigned) v195 * (unsigned) v9), v208, v209);
        TLOAD(v207, v210);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        bool v211 = v182 == v21;
        int32_t v212;
        if (v211) {
          v212 = v70;
        } else {
          v212 = v70;
        };
        event_t v213 = (event_t) v212;
        wait_flag(PIPE_M, PIPE_MTE1, v213);
        int64_t v214;
        if (v211) {
          v214 = v66;
        } else {
          v214 = v65;
        };
        Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v215;
        TASSIGN(v215, v214);
        int64_t v216;
        if (v193 == v21) {
          v216 = v66;
        } else {
          v216 = v65;
        };
        Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v217;
        TASSIGN(v217, v216);
        TMOV(v215, v217);
        Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v218;
        TASSIGN(v218, v214);
        Tile<TileType::Mat, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v219;
        TASSIGN(v219, v206);
        TMOV(v218, v219);
        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        event_t v220 = (event_t) v197;
        set_flag(PIPE_MTE1, PIPE_MTE2, v220);
        bool v221 = v183 == v21;
        int32_t v222;
        if (v221) {
          v222 = v70;
        } else {
          v222 = v70;
        };
        event_t v223 = (event_t) v222;
        wait_flag(PIPE_FIX, PIPE_M, v223);
        int64_t v224;
        if (v221) {
          v224 = v66;
        } else {
          v224 = v64;
        };
        Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v225;
        TASSIGN(v225, v224);
        Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v226;
        TASSIGN(v226, v214);
        Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v227;
        TASSIGN(v227, v214);
        TMATMUL(v225, v226, v227);
        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        event_t v228 = (event_t) v212;
        set_flag(PIPE_M, PIPE_MTE1, v228);
        Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v229;
        TASSIGN(v229, v224);
        unsigned v230 = (unsigned) v10;
        unsigned v231 = v15 * v230;
        pto::Shape<1, 1, 1, 128, 128> v232 = pto::Shape<1, 1, 1, 128, 128>();
        pto::Stride<-1, -1, -1, -1, 1> v233 = pto::Stride<-1, -1, -1, -1, 1>(v231, v231, v231, v230);
        GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v234 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v5 + (v17 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v194 * (uint32_t) v8) + (uint32_t) v95) * (unsigned) v10 + (unsigned) v195 * (unsigned) v18), v232, v233);
        TSTORE(v234, v229);
        event_t v235 = (event_t) v222;
        set_flag(PIPE_FIX, PIPE_M, v235);
        uint32_t v236 = (uint32_t) v182;
        uint32_t v237 = (uint32_t) v183;
        int32_t v238;
        if (v194 == v21) {
          v238 = v70;
        } else {
          v238 = v70;
        };
        bool v239 = v238 == v21;
        if (v239) {
          uint16_t v240 = getFFTSMsg(FFTS_MODE_VAL, v21);
          ffts_cross_core_sync(PIPE_FIX, v240);
        };
        bool v241 = v238 == v18;
        if (v241) {
          uint16_t v242 = getFFTSMsg(FFTS_MODE_VAL, v18);
          ffts_cross_core_sync(PIPE_FIX, v242);
        };
        bool v243 = v238 == v22;
        if (v243) {
          uint16_t v244 = getFFTSMsg(FFTS_MODE_VAL, v22);
          ffts_cross_core_sync(PIPE_FIX, v244);
        };
        bool v245 = v238 == v27;
        if (v245) {
          uint16_t v246 = getFFTSMsg(FFTS_MODE_VAL, v27);
          ffts_cross_core_sync(PIPE_FIX, v246);
        };
        bool v247 = v238 == v28;
        if (v247) {
          uint16_t v248 = getFFTSMsg(FFTS_MODE_VAL, v28);
          ffts_cross_core_sync(PIPE_FIX, v248);
        };
        bool v249 = v238 == v29;
        if (v249) {
          uint16_t v250 = getFFTSMsg(FFTS_MODE_VAL, v29);
          ffts_cross_core_sync(PIPE_FIX, v250);
        };
        bool v251 = v238 == v30;
        if (v251) {
          uint16_t v252 = getFFTSMsg(FFTS_MODE_VAL, v30);
          ffts_cross_core_sync(PIPE_FIX, v252);
        };
        bool v253 = v238 == v31;
        if (v253) {
          uint16_t v254 = getFFTSMsg(FFTS_MODE_VAL, v31);
          ffts_cross_core_sync(PIPE_FIX, v254);
        };
        bool v255 = v238 == v32;
        if (v255) {
          uint16_t v256 = getFFTSMsg(FFTS_MODE_VAL, v32);
          ffts_cross_core_sync(PIPE_FIX, v256);
        };
        bool v257 = v238 == v33;
        if (v257) {
          uint16_t v258 = getFFTSMsg(FFTS_MODE_VAL, v33);
          ffts_cross_core_sync(PIPE_FIX, v258);
        };
        bool v259 = v238 == v34;
        if (v259) {
          uint16_t v260 = getFFTSMsg(FFTS_MODE_VAL, v34);
          ffts_cross_core_sync(PIPE_FIX, v260);
        };
        bool v261 = v238 == v35;
        if (v261) {
          uint16_t v262 = getFFTSMsg(FFTS_MODE_VAL, v35);
          ffts_cross_core_sync(PIPE_FIX, v262);
        };
        bool v263 = v238 == v36;
        if (v263) {
          uint16_t v264 = getFFTSMsg(FFTS_MODE_VAL, v36);
          ffts_cross_core_sync(PIPE_FIX, v264);
        };
        bool v265 = v238 == v37;
        if (v265) {
          uint16_t v266 = getFFTSMsg(FFTS_MODE_VAL, v37);
          ffts_cross_core_sync(PIPE_FIX, v266);
        };
        bool v267 = v238 == v38;
        if (v267) {
          uint16_t v268 = getFFTSMsg(FFTS_MODE_VAL, v38);
          ffts_cross_core_sync(PIPE_FIX, v268);
        };
        bool v269 = v238 == v39;
        if (v269) {
          uint16_t v270 = getFFTSMsg(FFTS_MODE_VAL, v39);
          ffts_cross_core_sync(PIPE_FIX, v270);
        };
        v187 = (int32_t) ((size_t) v192);
        v188 = (int32_t) ((size_t) (int32_t) ((uint32_t) v18 - v236));
        v189 = (int32_t) ((size_t) (int32_t) ((uint32_t) v18 - v237));
        v190 = (int32_t) ((size_t) v186);
      } else {
        v187 = v181;
        v188 = v182;
        v189 = v183;
        v190 = v184;
      };
      uint32_t v271 = (uint32_t) v91;
      int32_t v272 = (int32_t) ((uint32_t) ((int32_t) v271 * (uint32_t) v81) + (uint32_t) v185) % v22;
      int32_t v273 = v91 % v22;
      int32_t v274 = v185 % v22;
      int32_t v275 = (int32_t) ((uint32_t) v185 * (uint32_t) v20);
      bool v276 = v272 == v21;
      int32_t v277;
      if (v276) {
        v277 = v62;
      } else {
        v277 = v62;
      };
      event_t v278 = (event_t) v277;
      wait_flag(PIPE_MTE1, PIPE_MTE2, v278);
      int64_t v279;
      if (v276) {
        v279 = (int64_t) ((int64_t) v40);
      } else {
        v279 = (int64_t) ((int64_t) v41);
      };
      Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v280;
      TASSIGN(v280, v279);
      unsigned v281 = (unsigned) v9;
      unsigned v282 = v15 * v281;
      pto::Shape<1, 1, 1, 128, 128> v283 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v284 = pto::Stride<-1, -1, -1, -1, 1>(v282, v282, v282, v281);
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v285 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v3 + (v17 + (unsigned) v275 * (unsigned) v9 + v17 * (unsigned) v18), v283, v284);
      TLOAD(v280, v285);
      bool v286 = v274 == v21;
      int32_t v287;
      if (v286) {
        v287 = v62;
      } else {
        v287 = v62;
      };
      bool v288 = v287 == v21;
      if (v288) {
        wait_flag_dev(0);
      };
      bool v289 = v287 == v18;
      if (v289) {
        wait_flag_dev(1);
      };
      bool v290 = v287 == v22;
      if (v290) {
        wait_flag_dev(2);
      };
      bool v291 = v287 == v27;
      if (v291) {
        wait_flag_dev(3);
      };
      bool v292 = v287 == v28;
      if (v292) {
        wait_flag_dev(4);
      };
      bool v293 = v287 == v29;
      if (v293) {
        wait_flag_dev(5);
      };
      bool v294 = v287 == v30;
      if (v294) {
        wait_flag_dev(6);
      };
      bool v295 = v287 == v31;
      if (v295) {
        wait_flag_dev(7);
      };
      bool v296 = v287 == v32;
      if (v296) {
        wait_flag_dev(8);
      };
      bool v297 = v287 == v33;
      if (v297) {
        wait_flag_dev(9);
      };
      bool v298 = v287 == v34;
      if (v298) {
        wait_flag_dev(10);
      };
      bool v299 = v287 == v35;
      if (v299) {
        wait_flag_dev(11);
      };
      bool v300 = v287 == v36;
      if (v300) {
        wait_flag_dev(12);
      };
      bool v301 = v287 == v37;
      if (v301) {
        wait_flag_dev(13);
      };
      bool v302 = v287 == v38;
      if (v302) {
        wait_flag_dev(14);
      };
      bool v303 = v287 == v39;
      if (v303) {
        wait_flag_dev(15);
      };
      int64_t v304;
      if (v276) {
        v304 = (int64_t) ((int64_t) v42);
      } else {
        v304 = (int64_t) ((int64_t) v43);
      };
      Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v305;
      TASSIGN(v305, v304);
      unsigned v306 = (unsigned) v10;
      unsigned v307 = v15 * v306;
      pto::Shape<1, 1, 1, 128, 128> v308 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v309 = pto::Stride<-1, -1, -1, -1, 1>(v307, v307, v307, v306);
      GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v310 = GlobalTensor<half, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v6 + (v17 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v274 * (uint32_t) v8) + (uint32_t) v95) * (unsigned) v10 + (unsigned) v275 * (unsigned) v18), v308, v309);
      TLOAD(v305, v310);
      set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
      bool v311 = v188 == v21;
      int32_t v312;
      if (v311) {
        v312 = v70;
      } else {
        v312 = v70;
      };
      event_t v313 = (event_t) v312;
      wait_flag(PIPE_M, PIPE_MTE1, v313);
      int64_t v314;
      if (v311) {
        v314 = v66;
      } else {
        v314 = v65;
      };
      Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v315;
      TASSIGN(v315, v314);
      Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v316;
      TASSIGN(v316, v304);
      TMOV(v315, v316);
      Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v317;
      TASSIGN(v317, v314);
      Tile<TileType::Mat, half, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v318;
      TASSIGN(v318, v279);
      TMOV(v317, v318);
      event_t v319 = (event_t) v277;
      set_flag(PIPE_MTE1, PIPE_MTE2, v319);
      set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
      bool v320 = v189 == v21;
      int32_t v321;
      if (v320) {
        v321 = v70;
      } else {
        v321 = v70;
      };
      event_t v322 = (event_t) v321;
      wait_flag(PIPE_FIX, PIPE_M, v322);
      int64_t v323;
      if (v320) {
        v323 = v66;
      } else {
        v323 = v64;
      };
      Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v324;
      TASSIGN(v324, v323);
      Tile<TileType::Left, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::RowMajor, 512, PadValue::Null> v325;
      TASSIGN(v325, v314);
      Tile<TileType::Right, half, 128, 128, BLayout::RowMajor, 128, 128, SLayout::ColMajor, 512, PadValue::Null> v326;
      TASSIGN(v326, v314);
      TMATMUL(v324, v325, v326);
      set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
      event_t v327 = (event_t) v312;
      set_flag(PIPE_M, PIPE_MTE1, v327);
      Tile<TileType::Acc, float, 128, 128, BLayout::ColMajor, 128, 128, SLayout::RowMajor, 1024, PadValue::Null> v328;
      TASSIGN(v328, v323);
      unsigned v329 = (unsigned) v9;
      unsigned v330 = v15 * v329;
      pto::Shape<1, 1, 1, 128, 128> v331 = pto::Shape<1, 1, 1, 128, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v332 = pto::Stride<-1, -1, -1, -1, 1>(v330, v330, v330, v329);
      GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v333 = GlobalTensor<float, pto::Shape<1, 1, 1, 128, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v17 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v86 * (uint32_t) v44) + (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v273 * (uint32_t) v22) * (uint32_t) v20)) + (uint32_t) ((int32_t) (uint32_t) v274 * (uint32_t) v20)) * (unsigned) v9 + v17 * (unsigned) v18), v331, v332);
      TSTORE(v333, v328);
      event_t v334 = (event_t) v321;
      set_flag(PIPE_FIX, PIPE_M, v334);
      uint32_t v335 = (uint32_t) v188;
      uint32_t v336 = (uint32_t) v189;
      int32_t v337;
      if (v286) {
        v337 = v61;
      } else {
        v337 = v61;
      };
      bool v338 = v337 == v21;
      if (v338) {
        uint16_t v339 = getFFTSMsg(FFTS_MODE_VAL, v21);
        ffts_cross_core_sync(PIPE_FIX, v339);
      };
      bool v340 = v337 == v18;
      if (v340) {
        uint16_t v341 = getFFTSMsg(FFTS_MODE_VAL, v18);
        ffts_cross_core_sync(PIPE_FIX, v341);
      };
      bool v342 = v337 == v22;
      if (v342) {
        uint16_t v343 = getFFTSMsg(FFTS_MODE_VAL, v22);
        ffts_cross_core_sync(PIPE_FIX, v343);
      };
      bool v344 = v337 == v27;
      if (v344) {
        uint16_t v345 = getFFTSMsg(FFTS_MODE_VAL, v27);
        ffts_cross_core_sync(PIPE_FIX, v345);
      };
      bool v346 = v337 == v28;
      if (v346) {
        uint16_t v347 = getFFTSMsg(FFTS_MODE_VAL, v28);
        ffts_cross_core_sync(PIPE_FIX, v347);
      };
      bool v348 = v337 == v29;
      if (v348) {
        uint16_t v349 = getFFTSMsg(FFTS_MODE_VAL, v29);
        ffts_cross_core_sync(PIPE_FIX, v349);
      };
      bool v350 = v337 == v30;
      if (v350) {
        uint16_t v351 = getFFTSMsg(FFTS_MODE_VAL, v30);
        ffts_cross_core_sync(PIPE_FIX, v351);
      };
      bool v352 = v337 == v31;
      if (v352) {
        uint16_t v353 = getFFTSMsg(FFTS_MODE_VAL, v31);
        ffts_cross_core_sync(PIPE_FIX, v353);
      };
      bool v354 = v337 == v32;
      if (v354) {
        uint16_t v355 = getFFTSMsg(FFTS_MODE_VAL, v32);
        ffts_cross_core_sync(PIPE_FIX, v355);
      };
      bool v356 = v337 == v33;
      if (v356) {
        uint16_t v357 = getFFTSMsg(FFTS_MODE_VAL, v33);
        ffts_cross_core_sync(PIPE_FIX, v357);
      };
      bool v358 = v337 == v34;
      if (v358) {
        uint16_t v359 = getFFTSMsg(FFTS_MODE_VAL, v34);
        ffts_cross_core_sync(PIPE_FIX, v359);
      };
      bool v360 = v337 == v35;
      if (v360) {
        uint16_t v361 = getFFTSMsg(FFTS_MODE_VAL, v35);
        ffts_cross_core_sync(PIPE_FIX, v361);
      };
      bool v362 = v337 == v36;
      if (v362) {
        uint16_t v363 = getFFTSMsg(FFTS_MODE_VAL, v36);
        ffts_cross_core_sync(PIPE_FIX, v363);
      };
      bool v364 = v337 == v37;
      if (v364) {
        uint16_t v365 = getFFTSMsg(FFTS_MODE_VAL, v37);
        ffts_cross_core_sync(PIPE_FIX, v365);
      };
      bool v366 = v337 == v38;
      if (v366) {
        uint16_t v367 = getFFTSMsg(FFTS_MODE_VAL, v38);
        ffts_cross_core_sync(PIPE_FIX, v367);
      };
      bool v368 = v337 == v39;
      if (v368) {
        uint16_t v369 = getFFTSMsg(FFTS_MODE_VAL, v39);
        ffts_cross_core_sync(PIPE_FIX, v369);
      };
      v181 = (int32_t) ((size_t) v272);
      v182 = (int32_t) ((size_t) (int32_t) ((uint32_t) v18 - v335));
      v183 = (int32_t) ((size_t) (int32_t) ((uint32_t) v18 - v336));
      v184 = v185;
    };
    uint32_t v370 = (uint32_t) v91;
    v88 = v181;
    v89 = v182;
    v90 = v183;
    v91 = (int32_t) ((size_t) (int32_t) (v370 + (uint32_t) v18));
    v92 = (int32_t) ((size_t) v95);
    v93 = v184;
  }
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID2);
  wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID3);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
  wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID0);
  wait_flag(PIPE_FIX, PIPE_M, EVENT_ID1);
  #endif // __DAV_CUBE__


  #if defined(__DAV_VEC__)
  set_mask_norm();
  set_vector_mask(-1, -1);
  int64_t v371 = get_subblockid();
  int32_t v372 = (int32_t) ((uint32_t) ((int32_t) (int64_t) v371) * (uint32_t) v45);
  int32_t v373;
  v373 = v68;
  for (int32_t v374 = v87; v374 < v80; v374 += v84) {
    int32_t v375 = (int32_t) ((uint32_t) v374 * (uint32_t) v20);
    int32_t v376 = v373 % v22;
    bool v377 = v376 == v21;
    int64_t v378;
    if (v377) {
      v378 = (int64_t) ((int64_t) v46);
    } else {
      v378 = (int64_t) ((int64_t) v47);
    };
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, 64, 1, SLayout::NoneBox, 512, PadValue::Null> v379;
    TASSIGN(v379, v378);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v380;
    TASSIGN(v380, v378);
    int64_t v381;
    if (v377) {
      v381 = (int64_t) ((int64_t) v48);
    } else {
      v381 = (int64_t) ((int64_t) v49);
    };
    Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, 64, 1, SLayout::NoneBox, 512, PadValue::Null> v382;
    TASSIGN(v382, v381);
    Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v383;
    TASSIGN(v383, v381);
    for (size_t v384 = v67; v384 < v69; v384 += v69) {
      int32_t v385 = (int32_t) v384;
      int32_t v386 = v385 % v22;
      int32_t v387 = (int32_t) ((uint32_t) v385 * (uint32_t) v20);
      bool v388 = v386 == v21;
      int32_t v389;
      if (v388) {
        v389 = v70;
      } else {
        v389 = v70;
      };
      bool v390 = v389 == v21;
      if (v390) {
        wait_flag_dev(0);
      };
      bool v391 = v389 == v18;
      if (v391) {
        wait_flag_dev(1);
      };
      bool v392 = v389 == v22;
      if (v392) {
        wait_flag_dev(2);
      };
      bool v393 = v389 == v27;
      if (v393) {
        wait_flag_dev(3);
      };
      bool v394 = v389 == v28;
      if (v394) {
        wait_flag_dev(4);
      };
      bool v395 = v389 == v29;
      if (v395) {
        wait_flag_dev(5);
      };
      bool v396 = v389 == v30;
      if (v396) {
        wait_flag_dev(6);
      };
      bool v397 = v389 == v31;
      if (v397) {
        wait_flag_dev(7);
      };
      bool v398 = v389 == v32;
      if (v398) {
        wait_flag_dev(8);
      };
      bool v399 = v389 == v33;
      if (v399) {
        wait_flag_dev(9);
      };
      bool v400 = v389 == v34;
      if (v400) {
        wait_flag_dev(10);
      };
      bool v401 = v389 == v35;
      if (v401) {
        wait_flag_dev(11);
      };
      bool v402 = v389 == v36;
      if (v402) {
        wait_flag_dev(12);
      };
      bool v403 = v389 == v37;
      if (v403) {
        wait_flag_dev(13);
      };
      bool v404 = v389 == v38;
      if (v404) {
        wait_flag_dev(14);
      };
      bool v405 = v389 == v39;
      if (v405) {
        wait_flag_dev(15);
      };
      int32_t v406 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v386 * (uint32_t) v8) + (uint32_t) v375) + (uint32_t) v372);
      unsigned v407 = (unsigned) v10;
      unsigned v408 = v14 * v407;
      pto::Shape<1, 1, 1, 64, 128> v409 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v410 = pto::Stride<-1, -1, -1, -1, 1>(v408, v408, v408, v407);
      GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v411 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v5 + (v17 + (unsigned) v406 * (unsigned) v10 + (unsigned) v387 * (unsigned) v18), v409, v410);
      TLOAD(v72, v411);
      set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      if (v385 == v21) {
        TROWMAX(v75, v72, v73);
        pipe_barrier(PIPE_V);
        TROWEXPANDSUB(v73, v72, v75);
        TMULS(v379, v75, v50);
        TMULS(v73, v73, v51);
        TEXP(v72, v73);
        pipe_barrier(PIPE_V);
        TROWSUM(v75, v72, v73);
        pipe_barrier(PIPE_V);
        TMULS(v382, v75, v50);
        TCVT(v74, v72, v13);
      };
      if (v385 > v21) {
        TROWMAX(v75, v72, v73);
        pipe_barrier(PIPE_V);
        TMAX(v76, v76, v380);
        pipe_barrier(PIPE_V);
        int64_t v412;
        if (v388) {
          v412 = v60;
        } else {
          v412 = v59;
        };
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v413;
        TASSIGN(v413, v412);
        TSUB(v413, v380, v76);
        pipe_barrier(PIPE_V);
        TMULS(v380, v76, v50);
        pipe_barrier(PIPE_V);
        TROWEXPANDSUB(v73, v72, v75);
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v414;
        TASSIGN(v414, v412);
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v415;
        TASSIGN(v415, v412);
        TMULS(v414, v415, v51);
        TMULS(v73, v73, v51);
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v416;
        TASSIGN(v416, v412);
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v417;
        TASSIGN(v417, v412);
        TEXP(v416, v417);
        TEXP(v72, v73);
        TCVT(v74, v72, v13);
        pipe_barrier(PIPE_V);
        Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v418;
        TASSIGN(v418, v412);
        TMUL(v383, v383, v418);
        TROWSUM(v75, v72, v73);
        pipe_barrier(PIPE_V);
        TADD(v383, v383, v76);
      };
      set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
      unsigned v419 = (unsigned) v10;
      unsigned v420 = v14 * v419;
      pto::Shape<1, 1, 1, 64, 128> v421 = pto::Shape<1, 1, 1, 64, 128>();
      pto::Stride<-1, -1, -1, -1, 1> v422 = pto::Stride<-1, -1, -1, -1, 1>(v420, v420, v420, v419);
      GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v423 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v6 + (v17 + (unsigned) v406 * (unsigned) v10 + (unsigned) v387 * (unsigned) v18), v421, v422);
      TSTORE(v423, v74);
      int32_t v424;
      if (v388) {
        v424 = v62;
      } else {
        v424 = v62;
      };
      bool v425 = v424 == v21;
      if (v425) {
        uint16_t v426 = getFFTSMsg(FFTS_MODE_VAL, v21);
        ffts_cross_core_sync(PIPE_MTE3, v426);
      };
      bool v427 = v424 == v18;
      if (v427) {
        uint16_t v428 = getFFTSMsg(FFTS_MODE_VAL, v18);
        ffts_cross_core_sync(PIPE_MTE3, v428);
      };
      bool v429 = v424 == v22;
      if (v429) {
        uint16_t v430 = getFFTSMsg(FFTS_MODE_VAL, v22);
        ffts_cross_core_sync(PIPE_MTE3, v430);
      };
      bool v431 = v424 == v27;
      if (v431) {
        uint16_t v432 = getFFTSMsg(FFTS_MODE_VAL, v27);
        ffts_cross_core_sync(PIPE_MTE3, v432);
      };
      bool v433 = v424 == v28;
      if (v433) {
        uint16_t v434 = getFFTSMsg(FFTS_MODE_VAL, v28);
        ffts_cross_core_sync(PIPE_MTE3, v434);
      };
      bool v435 = v424 == v29;
      if (v435) {
        uint16_t v436 = getFFTSMsg(FFTS_MODE_VAL, v29);
        ffts_cross_core_sync(PIPE_MTE3, v436);
      };
      bool v437 = v424 == v30;
      if (v437) {
        uint16_t v438 = getFFTSMsg(FFTS_MODE_VAL, v30);
        ffts_cross_core_sync(PIPE_MTE3, v438);
      };
      bool v439 = v424 == v31;
      if (v439) {
        uint16_t v440 = getFFTSMsg(FFTS_MODE_VAL, v31);
        ffts_cross_core_sync(PIPE_MTE3, v440);
      };
      bool v441 = v424 == v32;
      if (v441) {
        uint16_t v442 = getFFTSMsg(FFTS_MODE_VAL, v32);
        ffts_cross_core_sync(PIPE_MTE3, v442);
      };
      bool v443 = v424 == v33;
      if (v443) {
        uint16_t v444 = getFFTSMsg(FFTS_MODE_VAL, v33);
        ffts_cross_core_sync(PIPE_MTE3, v444);
      };
      bool v445 = v424 == v34;
      if (v445) {
        uint16_t v446 = getFFTSMsg(FFTS_MODE_VAL, v34);
        ffts_cross_core_sync(PIPE_MTE3, v446);
      };
      bool v447 = v424 == v35;
      if (v447) {
        uint16_t v448 = getFFTSMsg(FFTS_MODE_VAL, v35);
        ffts_cross_core_sync(PIPE_MTE3, v448);
      };
      bool v449 = v424 == v36;
      if (v449) {
        uint16_t v450 = getFFTSMsg(FFTS_MODE_VAL, v36);
        ffts_cross_core_sync(PIPE_MTE3, v450);
      };
      bool v451 = v424 == v37;
      if (v451) {
        uint16_t v452 = getFFTSMsg(FFTS_MODE_VAL, v37);
        ffts_cross_core_sync(PIPE_MTE3, v452);
      };
      bool v453 = v424 == v38;
      if (v453) {
        uint16_t v454 = getFFTSMsg(FFTS_MODE_VAL, v38);
        ffts_cross_core_sync(PIPE_MTE3, v454);
      };
      bool v455 = v424 == v39;
      if (v455) {
        uint16_t v456 = getFFTSMsg(FFTS_MODE_VAL, v39);
        ffts_cross_core_sync(PIPE_MTE3, v456);
      };
    };
    for (size_t v457 = v67; v457 < v82; v457 += v69) {
      int32_t v458 = (int32_t) v457;
      int32_t v459 = (int32_t) ((uint32_t) v458 + (uint32_t) v18);
      if (v459 < v81) {
        int32_t v460 = v459 % v22;
        int32_t v461 = (int32_t) ((uint32_t) v459 * (uint32_t) v20);
        bool v462 = v460 == v21;
        int32_t v463;
        if (v462) {
          v463 = v70;
        } else {
          v463 = v70;
        };
        bool v464 = v463 == v21;
        if (v464) {
          wait_flag_dev(0);
        };
        bool v465 = v463 == v18;
        if (v465) {
          wait_flag_dev(1);
        };
        bool v466 = v463 == v22;
        if (v466) {
          wait_flag_dev(2);
        };
        bool v467 = v463 == v27;
        if (v467) {
          wait_flag_dev(3);
        };
        bool v468 = v463 == v28;
        if (v468) {
          wait_flag_dev(4);
        };
        bool v469 = v463 == v29;
        if (v469) {
          wait_flag_dev(5);
        };
        bool v470 = v463 == v30;
        if (v470) {
          wait_flag_dev(6);
        };
        bool v471 = v463 == v31;
        if (v471) {
          wait_flag_dev(7);
        };
        bool v472 = v463 == v32;
        if (v472) {
          wait_flag_dev(8);
        };
        bool v473 = v463 == v33;
        if (v473) {
          wait_flag_dev(9);
        };
        bool v474 = v463 == v34;
        if (v474) {
          wait_flag_dev(10);
        };
        bool v475 = v463 == v35;
        if (v475) {
          wait_flag_dev(11);
        };
        bool v476 = v463 == v36;
        if (v476) {
          wait_flag_dev(12);
        };
        bool v477 = v463 == v37;
        if (v477) {
          wait_flag_dev(13);
        };
        bool v478 = v463 == v38;
        if (v478) {
          wait_flag_dev(14);
        };
        bool v479 = v463 == v39;
        if (v479) {
          wait_flag_dev(15);
        };
        int32_t v480 = (int32_t) ((uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v460 * (uint32_t) v8) + (uint32_t) v375) + (uint32_t) v372);
        unsigned v481 = (unsigned) v10;
        unsigned v482 = v14 * v481;
        pto::Shape<1, 1, 1, 64, 128> v483 = pto::Shape<1, 1, 1, 64, 128>();
        pto::Stride<-1, -1, -1, -1, 1> v484 = pto::Stride<-1, -1, -1, -1, 1>(v482, v482, v482, v481);
        GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v485 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v5 + (v17 + (unsigned) v480 * (unsigned) v10 + (unsigned) v461 * (unsigned) v18), v483, v484);
        TLOAD(v72, v485);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        if (v459 == v21) {
          TROWMAX(v75, v72, v73);
          pipe_barrier(PIPE_V);
          TROWEXPANDSUB(v73, v72, v75);
          TMULS(v379, v75, v50);
          TMULS(v73, v73, v51);
          TEXP(v72, v73);
          pipe_barrier(PIPE_V);
          TROWSUM(v75, v72, v73);
          pipe_barrier(PIPE_V);
          TMULS(v382, v75, v50);
          TCVT(v74, v72, v13);
        };
        if (v459 > v21) {
          TROWMAX(v75, v72, v73);
          pipe_barrier(PIPE_V);
          TMAX(v76, v76, v380);
          pipe_barrier(PIPE_V);
          int64_t v486;
          if (v462) {
            v486 = v60;
          } else {
            v486 = v59;
          };
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v487;
          TASSIGN(v487, v486);
          TSUB(v487, v380, v76);
          pipe_barrier(PIPE_V);
          TMULS(v380, v76, v50);
          pipe_barrier(PIPE_V);
          TROWEXPANDSUB(v73, v72, v75);
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v488;
          TASSIGN(v488, v486);
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v489;
          TASSIGN(v489, v486);
          TMULS(v488, v489, v51);
          TMULS(v73, v73, v51);
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v490;
          TASSIGN(v490, v486);
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v491;
          TASSIGN(v491, v486);
          TEXP(v490, v491);
          TEXP(v72, v73);
          TCVT(v74, v72, v13);
          pipe_barrier(PIPE_V);
          Tile<TileType::Vec, float, 1, 64, BLayout::RowMajor, 1, 64, SLayout::NoneBox, 512, PadValue::Null> v492;
          TASSIGN(v492, v486);
          TMUL(v383, v383, v492);
          TROWSUM(v75, v72, v73);
          pipe_barrier(PIPE_V);
          TADD(v383, v383, v76);
        };
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        unsigned v493 = (unsigned) v10;
        unsigned v494 = v14 * v493;
        pto::Shape<1, 1, 1, 64, 128> v495 = pto::Shape<1, 1, 1, 64, 128>();
        pto::Stride<-1, -1, -1, -1, 1> v496 = pto::Stride<-1, -1, -1, -1, 1>(v494, v494, v494, v493);
        GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v497 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v6 + (v17 + (unsigned) v480 * (unsigned) v10 + (unsigned) v461 * (unsigned) v18), v495, v496);
        TSTORE(v497, v74);
        int32_t v498;
        if (v462) {
          v498 = v62;
        } else {
          v498 = v62;
        };
        bool v499 = v498 == v21;
        if (v499) {
          uint16_t v500 = getFFTSMsg(FFTS_MODE_VAL, v21);
          ffts_cross_core_sync(PIPE_MTE3, v500);
        };
        bool v501 = v498 == v18;
        if (v501) {
          uint16_t v502 = getFFTSMsg(FFTS_MODE_VAL, v18);
          ffts_cross_core_sync(PIPE_MTE3, v502);
        };
        bool v503 = v498 == v22;
        if (v503) {
          uint16_t v504 = getFFTSMsg(FFTS_MODE_VAL, v22);
          ffts_cross_core_sync(PIPE_MTE3, v504);
        };
        bool v505 = v498 == v27;
        if (v505) {
          uint16_t v506 = getFFTSMsg(FFTS_MODE_VAL, v27);
          ffts_cross_core_sync(PIPE_MTE3, v506);
        };
        bool v507 = v498 == v28;
        if (v507) {
          uint16_t v508 = getFFTSMsg(FFTS_MODE_VAL, v28);
          ffts_cross_core_sync(PIPE_MTE3, v508);
        };
        bool v509 = v498 == v29;
        if (v509) {
          uint16_t v510 = getFFTSMsg(FFTS_MODE_VAL, v29);
          ffts_cross_core_sync(PIPE_MTE3, v510);
        };
        bool v511 = v498 == v30;
        if (v511) {
          uint16_t v512 = getFFTSMsg(FFTS_MODE_VAL, v30);
          ffts_cross_core_sync(PIPE_MTE3, v512);
        };
        bool v513 = v498 == v31;
        if (v513) {
          uint16_t v514 = getFFTSMsg(FFTS_MODE_VAL, v31);
          ffts_cross_core_sync(PIPE_MTE3, v514);
        };
        bool v515 = v498 == v32;
        if (v515) {
          uint16_t v516 = getFFTSMsg(FFTS_MODE_VAL, v32);
          ffts_cross_core_sync(PIPE_MTE3, v516);
        };
        bool v517 = v498 == v33;
        if (v517) {
          uint16_t v518 = getFFTSMsg(FFTS_MODE_VAL, v33);
          ffts_cross_core_sync(PIPE_MTE3, v518);
        };
        bool v519 = v498 == v34;
        if (v519) {
          uint16_t v520 = getFFTSMsg(FFTS_MODE_VAL, v34);
          ffts_cross_core_sync(PIPE_MTE3, v520);
        };
        bool v521 = v498 == v35;
        if (v521) {
          uint16_t v522 = getFFTSMsg(FFTS_MODE_VAL, v35);
          ffts_cross_core_sync(PIPE_MTE3, v522);
        };
        bool v523 = v498 == v36;
        if (v523) {
          uint16_t v524 = getFFTSMsg(FFTS_MODE_VAL, v36);
          ffts_cross_core_sync(PIPE_MTE3, v524);
        };
        bool v525 = v498 == v37;
        if (v525) {
          uint16_t v526 = getFFTSMsg(FFTS_MODE_VAL, v37);
          ffts_cross_core_sync(PIPE_MTE3, v526);
        };
        bool v527 = v498 == v38;
        if (v527) {
          uint16_t v528 = getFFTSMsg(FFTS_MODE_VAL, v38);
          ffts_cross_core_sync(PIPE_MTE3, v528);
        };
        bool v529 = v498 == v39;
        if (v529) {
          uint16_t v530 = getFFTSMsg(FFTS_MODE_VAL, v39);
          ffts_cross_core_sync(PIPE_MTE3, v530);
        };
      };
      int32_t v531 = v458 % v22;
      bool v532 = v531 == v21;
      int32_t v533;
      if (v532) {
        v533 = v61;
      } else {
        v533 = v61;
      };
      bool v534 = v533 == v21;
      if (v534) {
        wait_flag_dev(0);
      };
      bool v535 = v533 == v18;
      if (v535) {
        wait_flag_dev(1);
      };
      bool v536 = v533 == v22;
      if (v536) {
        wait_flag_dev(2);
      };
      bool v537 = v533 == v27;
      if (v537) {
        wait_flag_dev(3);
      };
      bool v538 = v533 == v28;
      if (v538) {
        wait_flag_dev(4);
      };
      bool v539 = v533 == v29;
      if (v539) {
        wait_flag_dev(5);
      };
      bool v540 = v533 == v30;
      if (v540) {
        wait_flag_dev(6);
      };
      bool v541 = v533 == v31;
      if (v541) {
        wait_flag_dev(7);
      };
      bool v542 = v533 == v32;
      if (v542) {
        wait_flag_dev(8);
      };
      bool v543 = v533 == v33;
      if (v543) {
        wait_flag_dev(9);
      };
      bool v544 = v533 == v34;
      if (v544) {
        wait_flag_dev(10);
      };
      bool v545 = v533 == v35;
      if (v545) {
        wait_flag_dev(11);
      };
      bool v546 = v533 == v36;
      if (v546) {
        wait_flag_dev(12);
      };
      bool v547 = v533 == v37;
      if (v547) {
        wait_flag_dev(13);
      };
      bool v548 = v533 == v38;
      if (v548) {
        wait_flag_dev(14);
      };
      bool v549 = v533 == v39;
      if (v549) {
        wait_flag_dev(15);
      };
      if (v458 == v21) {
        unsigned v550 = (unsigned) v9;
        unsigned v551 = v14 * v550;
        pto::Shape<1, 1, 1, 64, 128> v552 = pto::Shape<1, 1, 1, 64, 128>();
        pto::Stride<-1, -1, -1, -1, 1> v553 = pto::Stride<-1, -1, -1, -1, 1>(v551, v551, v551, v550);
        GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v554 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v17 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v86 * (uint32_t) v44) + (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v376 * (uint32_t) v22) * (uint32_t) v20)) + (uint32_t) ((int32_t) (uint32_t) v531 * (uint32_t) v20)) + (uint32_t) v372) * (unsigned) v9 + v17 * (unsigned) v18), v552, v553);
        TLOAD(v77, v554);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
      };
      if (v458 > v21) {
        unsigned v555 = (unsigned) v9;
        unsigned v556 = v14 * v555;
        pto::Shape<1, 1, 1, 64, 128> v557 = pto::Shape<1, 1, 1, 64, 128>();
        pto::Stride<-1, -1, -1, -1, 1> v558 = pto::Stride<-1, -1, -1, -1, 1>(v556, v556, v556, v555);
        GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v559 = GlobalTensor<float, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v7 + (v17 + (unsigned) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v86 * (uint32_t) v44) + (uint32_t) ((int32_t) (uint32_t) ((int32_t) (uint32_t) v376 * (uint32_t) v22) * (uint32_t) v20)) + (uint32_t) ((int32_t) (uint32_t) v531 * (uint32_t) v20)) + (uint32_t) v372) * (unsigned) v9 + v17 * (unsigned) v18), v557, v558);
        TLOAD(v78, v559);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        int64_t v560;
        if (v532) {
          v560 = v60;
        } else {
          v560 = v59;
        };
        Tile<TileType::Vec, float, 64, 1, BLayout::ColMajor, 64, 1, SLayout::NoneBox, 512, PadValue::Null> v561;
        TASSIGN(v561, v560);
        TROWEXPANDMUL(v77, v77, v561);
        TADD(v77, v77, v78);
      };
    };
    uint32_t v562 = (uint32_t) v373;
    TROWEXPANDDIV(v77, v77, v382);
    TCVT(v79, v77, v13);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    unsigned v563 = (unsigned) v9;
    unsigned v564 = v14 * v563;
    pto::Shape<1, 1, 1, 64, 128> v565 = pto::Shape<1, 1, 1, 64, 128>();
    pto::Stride<-1, -1, -1, -1, 1> v566 = pto::Stride<-1, -1, -1, -1, 1>(v564, v564, v564, v563);
    GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND> v567 = GlobalTensor<half, pto::Shape<1, 1, 1, 64, 128>, pto::Stride<-1, -1, -1, -1, 1>, pto::Layout::ND>(v4 + (v17 + (unsigned) ((int32_t) (uint32_t) v375 + (uint32_t) v372) * (unsigned) v9 + v17 * (unsigned) v18), v565, v566);
    TSTORE(v567, v79);
    v373 = (int32_t) ((size_t) (int32_t) (v562 + (uint32_t) v18));
  }
  #endif // __DAV_VEC__

  return;
}

