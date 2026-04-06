precision mediump float;
varying vec4 vColor;
varying float vDofFade;
void main() {
  vec2 c = gl_PointCoord - 0.5;
  float r = dot(c, c);

  // blur: 0 = sharp (hard circle), ~0.88 = max blur (soft gaussian-like)
  float blur = vDofFade;

  // Expanded discard: account for the scaled-up point size
  float discardR = 0.25 + blur * 0.1;
  if (r > discardR) discard;

  // Edge softness: sharp nodes have a tight edge, blurry ones fade from center
  // Sharp: smoothstep from 0.20 to 0.25 (crisp edge)
  // Blurred: smoothstep from ~0.02 to 0.25 (wide soft falloff)
  float innerR = mix(0.20, 0.02, blur);
  float soft = 1.0 - smoothstep(innerR, 0.25, r);

  gl_FragColor = vec4(vColor.rgb, vColor.a * soft);
}
