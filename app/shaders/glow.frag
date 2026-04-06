precision mediump float;
varying vec4 vColor;
varying float vDofFade;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = dot(c, c);
  if (r > 0.25) discard;
  // Widen the glow falloff for out-of-focus nodes
  float glowWidth = mix(0.2, 0.35, vDofFade);
  float glow = 1.0 - smoothstep(0.0, glowWidth, r);
  glow = glow * glow * glow; // steep cubic falloff
  gl_FragColor = vec4(vColor.rgb, vColor.a * glow * 0.15);
}
