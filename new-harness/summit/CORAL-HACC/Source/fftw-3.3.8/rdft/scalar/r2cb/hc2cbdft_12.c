/*
 * Copyright (c) 2003, 2007-14 Matteo Frigo
 * Copyright (c) 2003, 2007-14 Massachusetts Institute of Technology
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA
 *
 */

/* This file was automatically generated --- DO NOT EDIT */
/* Generated on Thu May 24 08:07:58 EDT 2018 */

#include "rdft/codelet-rdft.h"

#if defined(ARCH_PREFERS_FMA) || defined(ISA_EXTENSION_PREFERS_FMA)

/* Generated by: ../../../genfft/gen_hc2cdft.native -fma -compact -variables 4 -pipeline-latency 4 -sign 1 -n 12 -dif -name hc2cbdft_12 -include rdft/scalar/hc2cb.h */

/*
 * This function contains 142 FP additions, 68 FP multiplications,
 * (or, 96 additions, 22 multiplications, 46 fused multiply/add),
 * 55 stack variables, 2 constants, and 48 memory accesses
 */
#include "rdft/scalar/hc2cb.h"

static void hc2cbdft_12(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DK(KP866025403, +0.866025403784438646763723170752936183471402627);
     DK(KP500000000, +0.500000000000000000000000000000000000000000000);
     {
	  INT m;
	  for (m = mb, W = W + ((mb - 1) * 22); m < me; m = m + 1, Rp = Rp + ms, Ip = Ip + ms, Rm = Rm - ms, Im = Im - ms, W = W + 22, MAKE_VOLATILE_STRIDE(48, rs)) {
	       E Tv, TC, TD, T1L, T1M, T2y, Tb, T1Z, T1E, T2D, T1e, T1U, TY, T2o, T13;
	       E T18, T19, T1O, T1P, T2E, Tm, T1V, T1H, T2z, T1h, T20, TO, T2p;
	       {
		    E T1, T4, Tu, TS, Tp, Ts, Tt, TT, T6, T9, TB, TV, Tw, Tz, TA;
		    E TW;
		    {
			 E T2, T3, Tq, Tr;
			 T1 = Rp[0];
			 T2 = Rp[WS(rs, 4)];
			 T3 = Rm[WS(rs, 3)];
			 T4 = T2 + T3;
			 Tu = T2 - T3;
			 TS = FNMS(KP500000000, T4, T1);
			 Tp = Ip[0];
			 Tq = Ip[WS(rs, 4)];
			 Tr = Im[WS(rs, 3)];
			 Ts = Tq - Tr;
			 Tt = FNMS(KP500000000, Ts, Tp);
			 TT = Tr + Tq;
		    }
		    {
			 E T7, T8, Tx, Ty;
			 T6 = Rm[WS(rs, 5)];
			 T7 = Rm[WS(rs, 1)];
			 T8 = Rp[WS(rs, 2)];
			 T9 = T7 + T8;
			 TB = T7 - T8;
			 TV = FNMS(KP500000000, T9, T6);
			 Tw = Im[WS(rs, 5)];
			 Tx = Im[WS(rs, 1)];
			 Ty = Ip[WS(rs, 2)];
			 Tz = Tx - Ty;
			 TA = FNMS(KP500000000, Tz, Tw);
			 TW = Tx + Ty;
		    }
		    {
			 E T5, Ta, T1C, T1D;
			 Tv = FMA(KP866025403, Tu, Tt);
			 TC = FNMS(KP866025403, TB, TA);
			 TD = Tv + TC;
			 T1L = FNMS(KP866025403, Tu, Tt);
			 T1M = FMA(KP866025403, TB, TA);
			 T2y = T1L + T1M;
			 T5 = T1 + T4;
			 Ta = T6 + T9;
			 Tb = T5 + Ta;
			 T1Z = T5 - Ta;
			 T1C = FMA(KP866025403, TT, TS);
			 T1D = FNMS(KP866025403, TW, TV);
			 T1E = T1C + T1D;
			 T2D = T1C - T1D;
			 {
			      E T1c, T1d, TU, TX;
			      T1c = Tp + Ts;
			      T1d = Tw + Tz;
			      T1e = T1c - T1d;
			      T1U = T1c + T1d;
			      TU = FNMS(KP866025403, TT, TS);
			      TX = FMA(KP866025403, TW, TV);
			      TY = TU - TX;
			      T2o = TU + TX;
			 }
		    }
	       }
	       {
		    E Tc, Tf, TE, T12, TZ, T10, TH, T11, Th, Tk, TJ, T17, T14, T15, TM;
		    E T16;
		    {
			 E Td, Te, TF, TG;
			 Tc = Rp[WS(rs, 3)];
			 Td = Rm[WS(rs, 4)];
			 Te = Rm[0];
			 Tf = Td + Te;
			 TE = FNMS(KP500000000, Tf, Tc);
			 T12 = Td - Te;
			 TZ = Ip[WS(rs, 3)];
			 TF = Im[WS(rs, 4)];
			 TG = Im[0];
			 T10 = TF + TG;
			 TH = TF - TG;
			 T11 = FMA(KP500000000, T10, TZ);
		    }
		    {
			 E Ti, Tj, TK, TL;
			 Th = Rm[WS(rs, 2)];
			 Ti = Rp[WS(rs, 1)];
			 Tj = Rp[WS(rs, 5)];
			 Tk = Ti + Tj;
			 TJ = FNMS(KP500000000, Tk, Th);
			 T17 = Ti - Tj;
			 T14 = Im[WS(rs, 2)];
			 TK = Ip[WS(rs, 5)];
			 TL = Ip[WS(rs, 1)];
			 T15 = TK + TL;
			 TM = TK - TL;
			 T16 = FMA(KP500000000, T15, T14);
		    }
		    {
			 E Tg, Tl, T1F, T1G;
			 T13 = FMA(KP866025403, T12, T11);
			 T18 = FNMS(KP866025403, T17, T16);
			 T19 = T13 + T18;
			 T1O = FNMS(KP866025403, T12, T11);
			 T1P = FMA(KP866025403, T17, T16);
			 T2E = T1O + T1P;
			 Tg = Tc + Tf;
			 Tl = Th + Tk;
			 Tm = Tg + Tl;
			 T1V = Tg - Tl;
			 T1F = FNMS(KP866025403, TH, TE);
			 T1G = FNMS(KP866025403, TM, TJ);
			 T1H = T1F + T1G;
			 T2z = T1F - T1G;
			 {
			      E T1f, T1g, TI, TN;
			      T1f = TZ - T10;
			      T1g = T15 - T14;
			      T1h = T1f + T1g;
			      T20 = T1f - T1g;
			      TI = FMA(KP866025403, TH, TE);
			      TN = FMA(KP866025403, TM, TJ);
			      TO = TI - TN;
			      T2p = TI + TN;
			 }
		    }
	       }
	       {
		    E Tn, T1i, TP, T1a, TQ, T1j, To, T1b, T1k, TR;
		    Tn = Tb + Tm;
		    T1i = T1e + T1h;
		    TP = TD + TO;
		    T1a = TY - T19;
		    To = W[0];
		    TQ = To * TP;
		    T1j = To * T1a;
		    TR = W[1];
		    T1b = FMA(TR, T1a, TQ);
		    T1k = FNMS(TR, TP, T1j);
		    Rp[0] = Tn - T1b;
		    Ip[0] = T1i + T1k;
		    Rm[0] = Tn + T1b;
		    Im[0] = T1k - T1i;
	       }
	       {
		    E T1p, T1l, T1n, T1o, T1x, T1s, T1v, T1t, T1z, T1m, T1r;
		    T1p = T1e - T1h;
		    T1m = Tb - Tm;
		    T1l = W[10];
		    T1n = T1l * T1m;
		    T1o = W[11];
		    T1x = T1o * T1m;
		    T1s = TD - TO;
		    T1v = TY + T19;
		    T1r = W[12];
		    T1t = T1r * T1s;
		    T1z = T1r * T1v;
		    {
			 E T1q, T1y, T1w, T1A, T1u;
			 T1q = FNMS(T1o, T1p, T1n);
			 T1y = FMA(T1l, T1p, T1x);
			 T1u = W[13];
			 T1w = FMA(T1u, T1v, T1t);
			 T1A = FNMS(T1u, T1s, T1z);
			 Rp[WS(rs, 3)] = T1q - T1w;
			 Ip[WS(rs, 3)] = T1y + T1A;
			 Rm[WS(rs, 3)] = T1q + T1w;
			 Im[WS(rs, 3)] = T1A - T1y;
		    }
	       }
	       {
		    E T1R, T2b, T27, T29, T2a, T2l, T1B, T1J, T1K, T25, T1W, T21, T1X, T23, T2e;
		    E T2h, T2f, T2j;
		    {
			 E T1N, T1Q, T28, T1I, T1T, T2d;
			 T1N = T1L - T1M;
			 T1Q = T1O - T1P;
			 T1R = T1N - T1Q;
			 T2b = T1N + T1Q;
			 T28 = T1E + T1H;
			 T27 = W[14];
			 T29 = T27 * T28;
			 T2a = W[15];
			 T2l = T2a * T28;
			 T1I = T1E - T1H;
			 T1B = W[2];
			 T1J = T1B * T1I;
			 T1K = W[3];
			 T25 = T1K * T1I;
			 T1W = T1U - T1V;
			 T21 = T1Z + T20;
			 T1T = W[4];
			 T1X = T1T * T1W;
			 T23 = T1T * T21;
			 T2e = T1V + T1U;
			 T2h = T1Z - T20;
			 T2d = W[16];
			 T2f = T2d * T2e;
			 T2j = T2d * T2h;
		    }
		    {
			 E T1S, T26, T22, T24, T1Y;
			 T1S = FNMS(T1K, T1R, T1J);
			 T26 = FMA(T1B, T1R, T25);
			 T1Y = W[5];
			 T22 = FMA(T1Y, T21, T1X);
			 T24 = FNMS(T1Y, T1W, T23);
			 Rp[WS(rs, 1)] = T1S - T22;
			 Ip[WS(rs, 1)] = T24 + T26;
			 Rm[WS(rs, 1)] = T22 + T1S;
			 Im[WS(rs, 1)] = T24 - T26;
		    }
		    {
			 E T2c, T2m, T2i, T2k, T2g;
			 T2c = FNMS(T2a, T2b, T29);
			 T2m = FMA(T27, T2b, T2l);
			 T2g = W[17];
			 T2i = FMA(T2g, T2h, T2f);
			 T2k = FNMS(T2g, T2e, T2j);
			 Rp[WS(rs, 4)] = T2c - T2i;
			 Ip[WS(rs, 4)] = T2k + T2m;
			 Rm[WS(rs, 4)] = T2i + T2c;
			 Im[WS(rs, 4)] = T2k - T2m;
		    }
	       }
	       {
		    E T2v, T2P, T2L, T2N, T2O, T2X, T2n, T2r, T2s, T2H, T2A, T2F, T2B, T2J, T2S;
		    E T2V, T2T, T2Z;
		    {
			 E T2t, T2u, T2M, T2q, T2x, T2R;
			 T2t = Tv - TC;
			 T2u = T13 - T18;
			 T2v = T2t + T2u;
			 T2P = T2t - T2u;
			 T2M = T2o - T2p;
			 T2L = W[18];
			 T2N = T2L * T2M;
			 T2O = W[19];
			 T2X = T2O * T2M;
			 T2q = T2o + T2p;
			 T2n = W[6];
			 T2r = T2n * T2q;
			 T2s = W[7];
			 T2H = T2s * T2q;
			 T2A = T2y + T2z;
			 T2F = T2D - T2E;
			 T2x = W[8];
			 T2B = T2x * T2A;
			 T2J = T2x * T2F;
			 T2S = T2y - T2z;
			 T2V = T2D + T2E;
			 T2R = W[20];
			 T2T = T2R * T2S;
			 T2Z = T2R * T2V;
		    }
		    {
			 E T2w, T2I, T2G, T2K, T2C;
			 T2w = FNMS(T2s, T2v, T2r);
			 T2I = FMA(T2n, T2v, T2H);
			 T2C = W[9];
			 T2G = FMA(T2C, T2F, T2B);
			 T2K = FNMS(T2C, T2A, T2J);
			 Rp[WS(rs, 2)] = T2w - T2G;
			 Ip[WS(rs, 2)] = T2I + T2K;
			 Rm[WS(rs, 2)] = T2w + T2G;
			 Im[WS(rs, 2)] = T2K - T2I;
		    }
		    {
			 E T2Q, T2Y, T2W, T30, T2U;
			 T2Q = FNMS(T2O, T2P, T2N);
			 T2Y = FMA(T2L, T2P, T2X);
			 T2U = W[21];
			 T2W = FMA(T2U, T2V, T2T);
			 T30 = FNMS(T2U, T2S, T2Z);
			 Rp[WS(rs, 5)] = T2Q - T2W;
			 Ip[WS(rs, 5)] = T2Y + T30;
			 Rm[WS(rs, 5)] = T2Q + T2W;
			 Im[WS(rs, 5)] = T30 - T2Y;
		    }
	       }
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 1, 12},
     {TW_NEXT, 1, 0}
};

static const hc2c_desc desc = { 12, "hc2cbdft_12", twinstr, &GENUS, {96, 22, 46, 0} };

void X(codelet_hc2cbdft_12) (planner *p) {
     X(khc2c_register) (p, hc2cbdft_12, &desc, HC2C_VIA_DFT);
}
#else

/* Generated by: ../../../genfft/gen_hc2cdft.native -compact -variables 4 -pipeline-latency 4 -sign 1 -n 12 -dif -name hc2cbdft_12 -include rdft/scalar/hc2cb.h */

/*
 * This function contains 142 FP additions, 60 FP multiplications,
 * (or, 112 additions, 30 multiplications, 30 fused multiply/add),
 * 47 stack variables, 2 constants, and 48 memory accesses
 */
#include "rdft/scalar/hc2cb.h"

static void hc2cbdft_12(R *Rp, R *Ip, R *Rm, R *Im, const R *W, stride rs, INT mb, INT me, INT ms)
{
     DK(KP500000000, +0.500000000000000000000000000000000000000000000);
     DK(KP866025403, +0.866025403784438646763723170752936183471402627);
     {
	  INT m;
	  for (m = mb, W = W + ((mb - 1) * 22); m < me; m = m + 1, Rp = Rp + ms, Ip = Ip + ms, Rm = Rm - ms, Im = Im - ms, W = W + 22, MAKE_VOLATILE_STRIDE(48, rs)) {
	       E Tv, T1E, TC, T1F, TW, T1x, TT, T1w, T1d, T1N, Tb, T1R, TI, T1z, TN;
	       E T1A, T17, T1I, T12, T1H, T1g, T1S, Tm, T1O;
	       {
		    E T1, Tq, T6, TA, T4, Tp, Tt, TS, T9, Tw, Tz, TV;
		    T1 = Rp[0];
		    Tq = Ip[0];
		    T6 = Rm[WS(rs, 5)];
		    TA = Im[WS(rs, 5)];
		    {
			 E T2, T3, Tr, Ts;
			 T2 = Rp[WS(rs, 4)];
			 T3 = Rm[WS(rs, 3)];
			 T4 = T2 + T3;
			 Tp = KP866025403 * (T2 - T3);
			 Tr = Im[WS(rs, 3)];
			 Ts = Ip[WS(rs, 4)];
			 Tt = Tr - Ts;
			 TS = KP866025403 * (Tr + Ts);
		    }
		    {
			 E T7, T8, Tx, Ty;
			 T7 = Rm[WS(rs, 1)];
			 T8 = Rp[WS(rs, 2)];
			 T9 = T7 + T8;
			 Tw = KP866025403 * (T7 - T8);
			 Tx = Im[WS(rs, 1)];
			 Ty = Ip[WS(rs, 2)];
			 Tz = Tx - Ty;
			 TV = KP866025403 * (Tx + Ty);
		    }
		    {
			 E Tu, TB, TU, TR;
			 Tu = FMA(KP500000000, Tt, Tq);
			 Tv = Tp + Tu;
			 T1E = Tu - Tp;
			 TB = FMS(KP500000000, Tz, TA);
			 TC = Tw + TB;
			 T1F = TB - Tw;
			 TU = FNMS(KP500000000, T9, T6);
			 TW = TU + TV;
			 T1x = TU - TV;
			 TR = FNMS(KP500000000, T4, T1);
			 TT = TR - TS;
			 T1w = TR + TS;
			 {
			      E T1b, T1c, T5, Ta;
			      T1b = Tq - Tt;
			      T1c = Tz + TA;
			      T1d = T1b - T1c;
			      T1N = T1b + T1c;
			      T5 = T1 + T4;
			      Ta = T6 + T9;
			      Tb = T5 + Ta;
			      T1R = T5 - Ta;
			 }
		    }
	       }
	       {
		    E Tc, T10, Th, T15, Tf, TY, TH, TZ, Tk, T13, TM, T14;
		    Tc = Rp[WS(rs, 3)];
		    T10 = Ip[WS(rs, 3)];
		    Th = Rm[WS(rs, 2)];
		    T15 = Im[WS(rs, 2)];
		    {
			 E Td, Te, TF, TG;
			 Td = Rm[WS(rs, 4)];
			 Te = Rm[0];
			 Tf = Td + Te;
			 TY = KP866025403 * (Td - Te);
			 TF = Im[WS(rs, 4)];
			 TG = Im[0];
			 TH = KP866025403 * (TF - TG);
			 TZ = TF + TG;
		    }
		    {
			 E Ti, Tj, TK, TL;
			 Ti = Rp[WS(rs, 1)];
			 Tj = Rp[WS(rs, 5)];
			 Tk = Ti + Tj;
			 T13 = KP866025403 * (Ti - Tj);
			 TK = Ip[WS(rs, 5)];
			 TL = Ip[WS(rs, 1)];
			 TM = KP866025403 * (TK - TL);
			 T14 = TK + TL;
		    }
		    {
			 E TE, TJ, T16, T11;
			 TE = FNMS(KP500000000, Tf, Tc);
			 TI = TE + TH;
			 T1z = TE - TH;
			 TJ = FNMS(KP500000000, Tk, Th);
			 TN = TJ + TM;
			 T1A = TJ - TM;
			 T16 = FMA(KP500000000, T14, T15);
			 T17 = T13 - T16;
			 T1I = T13 + T16;
			 T11 = FMA(KP500000000, TZ, T10);
			 T12 = TY + T11;
			 T1H = T11 - TY;
			 {
			      E T1e, T1f, Tg, Tl;
			      T1e = T10 - TZ;
			      T1f = T14 - T15;
			      T1g = T1e + T1f;
			      T1S = T1e - T1f;
			      Tg = Tc + Tf;
			      Tl = Th + Tk;
			      Tm = Tg + Tl;
			      T1O = Tg - Tl;
			 }
		    }
	       }
	       {
		    E Tn, T1h, TP, T1p, T19, T1r, T1n, T1t;
		    Tn = Tb + Tm;
		    T1h = T1d + T1g;
		    {
			 E TD, TO, TX, T18;
			 TD = Tv - TC;
			 TO = TI - TN;
			 TP = TD + TO;
			 T1p = TD - TO;
			 TX = TT - TW;
			 T18 = T12 - T17;
			 T19 = TX - T18;
			 T1r = TX + T18;
			 {
			      E T1k, T1m, T1j, T1l;
			      T1k = Tb - Tm;
			      T1m = T1d - T1g;
			      T1j = W[10];
			      T1l = W[11];
			      T1n = FNMS(T1l, T1m, T1j * T1k);
			      T1t = FMA(T1l, T1k, T1j * T1m);
			 }
		    }
		    {
			 E T1a, T1i, To, TQ;
			 To = W[0];
			 TQ = W[1];
			 T1a = FMA(To, TP, TQ * T19);
			 T1i = FNMS(TQ, TP, To * T19);
			 Rp[0] = Tn - T1a;
			 Ip[0] = T1h + T1i;
			 Rm[0] = Tn + T1a;
			 Im[0] = T1i - T1h;
		    }
		    {
			 E T1s, T1u, T1o, T1q;
			 T1o = W[12];
			 T1q = W[13];
			 T1s = FMA(T1o, T1p, T1q * T1r);
			 T1u = FNMS(T1q, T1p, T1o * T1r);
			 Rp[WS(rs, 3)] = T1n - T1s;
			 Ip[WS(rs, 3)] = T1t + T1u;
			 Rm[WS(rs, 3)] = T1n + T1s;
			 Im[WS(rs, 3)] = T1u - T1t;
		    }
	       }
	       {
		    E T1C, T1Y, T1K, T20, T1U, T1V, T26, T27;
		    {
			 E T1y, T1B, T1G, T1J;
			 T1y = T1w + T1x;
			 T1B = T1z + T1A;
			 T1C = T1y - T1B;
			 T1Y = T1y + T1B;
			 T1G = T1E + T1F;
			 T1J = T1H - T1I;
			 T1K = T1G - T1J;
			 T20 = T1G + T1J;
		    }
		    {
			 E T1P, T1T, T1M, T1Q;
			 T1P = T1N - T1O;
			 T1T = T1R + T1S;
			 T1M = W[4];
			 T1Q = W[5];
			 T1U = FMA(T1M, T1P, T1Q * T1T);
			 T1V = FNMS(T1Q, T1P, T1M * T1T);
		    }
		    {
			 E T23, T25, T22, T24;
			 T23 = T1O + T1N;
			 T25 = T1R - T1S;
			 T22 = W[16];
			 T24 = W[17];
			 T26 = FMA(T22, T23, T24 * T25);
			 T27 = FNMS(T24, T23, T22 * T25);
		    }
		    {
			 E T1L, T1W, T1v, T1D;
			 T1v = W[2];
			 T1D = W[3];
			 T1L = FNMS(T1D, T1K, T1v * T1C);
			 T1W = FMA(T1D, T1C, T1v * T1K);
			 Rp[WS(rs, 1)] = T1L - T1U;
			 Ip[WS(rs, 1)] = T1V + T1W;
			 Rm[WS(rs, 1)] = T1U + T1L;
			 Im[WS(rs, 1)] = T1V - T1W;
		    }
		    {
			 E T21, T28, T1X, T1Z;
			 T1X = W[14];
			 T1Z = W[15];
			 T21 = FNMS(T1Z, T20, T1X * T1Y);
			 T28 = FMA(T1Z, T1Y, T1X * T20);
			 Rp[WS(rs, 4)] = T21 - T26;
			 Ip[WS(rs, 4)] = T27 + T28;
			 Rm[WS(rs, 4)] = T26 + T21;
			 Im[WS(rs, 4)] = T27 - T28;
		    }
	       }
	       {
		    E T2c, T2u, T2p, T2B, T2g, T2w, T2l, T2z;
		    {
			 E T2a, T2b, T2n, T2o;
			 T2a = TT + TW;
			 T2b = TI + TN;
			 T2c = T2a + T2b;
			 T2u = T2a - T2b;
			 T2n = T1w - T1x;
			 T2o = T1H + T1I;
			 T2p = T2n - T2o;
			 T2B = T2n + T2o;
		    }
		    {
			 E T2e, T2f, T2j, T2k;
			 T2e = Tv + TC;
			 T2f = T12 + T17;
			 T2g = T2e + T2f;
			 T2w = T2e - T2f;
			 T2j = T1E - T1F;
			 T2k = T1z - T1A;
			 T2l = T2j + T2k;
			 T2z = T2j - T2k;
		    }
		    {
			 E T2h, T2r, T2q, T2s;
			 {
			      E T29, T2d, T2i, T2m;
			      T29 = W[6];
			      T2d = W[7];
			      T2h = FNMS(T2d, T2g, T29 * T2c);
			      T2r = FMA(T2d, T2c, T29 * T2g);
			      T2i = W[8];
			      T2m = W[9];
			      T2q = FMA(T2i, T2l, T2m * T2p);
			      T2s = FNMS(T2m, T2l, T2i * T2p);
			 }
			 Rp[WS(rs, 2)] = T2h - T2q;
			 Ip[WS(rs, 2)] = T2r + T2s;
			 Rm[WS(rs, 2)] = T2h + T2q;
			 Im[WS(rs, 2)] = T2s - T2r;
		    }
		    {
			 E T2x, T2D, T2C, T2E;
			 {
			      E T2t, T2v, T2y, T2A;
			      T2t = W[18];
			      T2v = W[19];
			      T2x = FNMS(T2v, T2w, T2t * T2u);
			      T2D = FMA(T2v, T2u, T2t * T2w);
			      T2y = W[20];
			      T2A = W[21];
			      T2C = FMA(T2y, T2z, T2A * T2B);
			      T2E = FNMS(T2A, T2z, T2y * T2B);
			 }
			 Rp[WS(rs, 5)] = T2x - T2C;
			 Ip[WS(rs, 5)] = T2D + T2E;
			 Rm[WS(rs, 5)] = T2x + T2C;
			 Im[WS(rs, 5)] = T2E - T2D;
		    }
	       }
	  }
     }
}

static const tw_instr twinstr[] = {
     {TW_FULL, 1, 12},
     {TW_NEXT, 1, 0}
};

static const hc2c_desc desc = { 12, "hc2cbdft_12", twinstr, &GENUS, {112, 30, 30, 0} };

void X(codelet_hc2cbdft_12) (planner *p) {
     X(khc2c_register) (p, hc2cbdft_12, &desc, HC2C_VIA_DFT);
}
#endif
