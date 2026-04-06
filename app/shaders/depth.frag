#extension GL_EXT_frag_depth : enable
precision mediump float;
varying float vClipZ;
varying float vClipW;
varying float vWorldRadius;
uniform float uNear;
uniform float uFar;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r2 = dot(c, c);
  if (r2 > 0.25) discard;

  // Sphere surface offset: normalized [0..1] from edge to center
  float sphereNorm = sqrt(0.25 - r2) * 2.0; // 0 at edge, 1 at center

  // Center depth in NDC [-1, 1]
  float ndcZ = vClipZ / vClipW;

  // Recover view-space Z (negative, looking down -Z)
  float fn2 = 2.0 * uFar * uNear;
  float fmn = uFar - uNear;
  float fpn = uFar + uNear;
  float zView = fn2 / (ndcZ * fmn - fpn);

  // Offset toward camera by sphere surface depth (view-space radius * normalized offset)
  float zAdjusted = zView + sphereNorm * vWorldRadius;

  // Convert back to NDC then to [0,1] depth
  float ndcAdj = (fpn + fn2 / zAdjusted) / fmn;
  gl_FragDepthEXT = (ndcAdj + 1.0) * 0.5;
  gl_FragColor = vec4(0.0);
}
