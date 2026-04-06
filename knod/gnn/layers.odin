package gnn

import "core:math"

linear_forward :: proc(out, inp, weight, bias: []f32, N, C_in, C_out: int) {
	for n in 0 ..< N {
		for o in 0 ..< C_out {
			val: f32 = 0
			if len(bias) > 0 {val = bias[o]}
			for i in 0 ..< C_in {
				val += inp[n * C_in + i] * weight[o * C_in + i]
			}
			out[n * C_out + o] = val
		}
	}
}

linear_backward :: proc(dinp, dweight, dbias, dout, inp, weight: []f32, N, C_in, C_out: int) {
	for n in 0 ..< N {
		for o in 0 ..< C_out {
			d := dout[n * C_out + o]
			wrow := weight[o * C_in:]
			for i in 0 ..< C_in {
				dinp[n * C_in + i] += wrow[i] * d
			}
		}
	}
	for o in 0 ..< C_out {
		for n in 0 ..< N {
			d := dout[n * C_out + o]
			if len(dbias) > 0 {dbias[o] += d}
			for i in 0 ..< C_in {
				dweight[o * C_in + i] += inp[n * C_in + i] * d
			}
		}
	}
}


relu_forward :: proc(out, inp: []f32, N: int) {
	for i in 0 ..< N {
		out[i] = max(inp[i], 0)
	}
}

relu_backward :: proc(dinp, dout, inp: []f32, N: int) {
	for i in 0 ..< N {
		if inp[i] > 0 {
			dinp[i] += dout[i]
		}
	}
}


layernorm_forward :: proc(out, mean, rstd, inp, weight, bias: []f32, N, C: int) {
	eps: f32 = 1e-5
	for n in 0 ..< N {
		x := inp[n * C:]
		m: f32 = 0
		for i in 0 ..< C {m += x[i]}
		m /= f32(C)
		v: f32 = 0
		for i in 0 ..< C {
			d := x[i] - m
			v += d * d
		}
		v /= f32(C)
		s := 1.0 / math.sqrt(v + eps)
		o := out[n * C:]
		for i in 0 ..< C {
			o[i] = (x[i] - m) * s * weight[i] + bias[i]
		}
		mean[n] = m
		rstd[n] = s
	}
}

layernorm_backward :: proc(dinp, dweight, dbias, dout, inp, weight, mean, rstd: []f32, N, C: int) {
	for n in 0 ..< N {
		dout_n := dout[n * C:]
		inp_n := inp[n * C:]
		dinp_n := dinp[n * C:]
		m := mean[n]
		s := rstd[n]

		dnorm_mean: f32 = 0
		dnorm_norm_mean: f32 = 0
		for i in 0 ..< C {
			norm_i := (inp_n[i] - m) * s
			dnorm_i := weight[i] * dout_n[i]
			dnorm_mean += dnorm_i
			dnorm_norm_mean += dnorm_i * norm_i
		}
		dnorm_mean /= f32(C)
		dnorm_norm_mean /= f32(C)

		for i in 0 ..< C {
			norm_i := (inp_n[i] - m) * s
			dnorm_i := weight[i] * dout_n[i]
			dbias[i] += dout_n[i]
			dweight[i] += norm_i * dout_n[i]
			dval := dnorm_i - dnorm_mean - norm_i * dnorm_norm_mean
			dval *= s
			dinp_n[i] += dval
		}
	}
}

concat3_forward :: proc(out: []f32, a, b, c: []f32, N, C: int) {
	for n in 0 ..< N {
		o := out[n * 3 * C:]
		for i in 0 ..< C {o[i] = a[n * C + i]}
		for i in 0 ..< C {o[C + i] = b[n * C + i]}
		for i in 0 ..< C {o[2 * C + i] = c[n * C + i]}
	}
}

concat3_backward :: proc(da, db, dc: []f32, dout: []f32, N, C: int) {
	for n in 0 ..< N {
		d := dout[n * 3 * C:]
		for i in 0 ..< C {da[n * C + i] += d[i]}
		for i in 0 ..< C {db[n * C + i] += d[C + i]}
		for i in 0 ..< C {dc[n * C + i] += d[2 * C + i]}
	}
}

concat2_forward :: proc(out: []f32, a, b: []f32, N, C: int) {
	for n in 0 ..< N {
		o := out[n * 2 * C:]
		for i in 0 ..< C {o[i] = a[n * C + i]}
		for i in 0 ..< C {o[C + i] = b[n * C + i]}
	}
}

concat2_backward :: proc(da, db: []f32, dout: []f32, N, C: int) {
	for n in 0 ..< N {
		d := dout[n * 2 * C:]
		for i in 0 ..< C {da[n * C + i] += d[i]}
		for i in 0 ..< C {db[n * C + i] += d[C + i]}
	}
}

scatter_add :: proc(out: []f32, messages: []f32, edge_dst: []int, num_edges, H: int) {
	for e in 0 ..< num_edges {
		dst := edge_dst[e]
		for i in 0 ..< H {
			out[dst * H + i] += messages[e * H + i]
		}
	}
}

scatter_add_backward :: proc(dmessages: []f32, dout: []f32, edge_dst: []int, num_edges, H: int) {
	for e in 0 ..< num_edges {
		dst := edge_dst[e]
		for i in 0 ..< H {
			dmessages[e * H + i] += dout[dst * H + i]
		}
	}
}
