//! Per-keyframe waves: a damped oscillation or bounce laid on top of a tween.
//!
//! A single temporal Bezier has at most one bend per side, so bounce and elastic shapes cannot be
//! drawn with eases alone. A wave is stored on the key it belongs to instead, so the property keeps
//! its two keyframes and the Graph Editor draws the combined curve like any other value.
//!
//! * `wave_after` on key *i* plays after the value arrives at key *i* and runs until the next key
//!   (fading out over the last 30 % of that segment so the next tween starts exactly on its key).
//!   Its size is a fraction of the travel from key *i − 1* to key *i*.
//! * `wave_before` on key *m* plays before the value leaves key *m*, after the previous key (fading
//!   in over the first 30 %). Its size is a fraction of the travel from key *m* to key *m + 1*.
//!
//! Only scalar, 2D and 3D values take a wave (colours and other kinds ignore it). On a spatial
//! property the wave swings along the straight line between the two keys.

use std::borrow::Borrow;

use effectcraft_time::Tick;
use serde::{Deserialize, Serialize};

use crate::{Keyframe, Value, secs};

/// Shape of a wave.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WaveKind {
    /// A damped sine: swings past the key and back.
    #[default]
    Oscillate,
    /// A ball bouncing off the key: each rebound is shorter and lower.
    Bounce,
}

impl WaveKind {
    pub fn label(self) -> &'static str {
        match self {
            WaveKind::Oscillate => "Oscillate",
            WaveKind::Bounce => "Bounce",
        }
    }
}

/// A damped wave laid on a key.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Wave {
    pub kind: WaveKind,
    /// Peak swing as a fraction of the distance travelled (0.15 = 15 %).
    pub size: f64,
    /// Oscillate: swings per second. Bounce: 1 / duration of the first bounce, in seconds.
    pub freq: f64,
    /// How fast the wave dies out; higher settles sooner.
    pub decay: f64,
}

impl Default for Wave {
    fn default() -> Self {
        Wave { kind: WaveKind::Oscillate, size: 0.15, freq: 3.0, decay: 5.0 }
    }
}

const MAX_BOUNCES: usize = 64;

impl Wave {
    /// A copy with every number finite and inside the range the evaluator accepts.
    pub fn sanitized(self) -> Wave {
        let fix = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.clamp(lo, hi) } else { or };
        let d = Wave::default();
        Wave { kind: self.kind, size: fix(self.size, -4.0, 4.0, d.size), freq: fix(self.freq, 0.05, 60.0, d.freq), decay: fix(self.decay, 0.0, 60.0, d.decay) }
    }

    /// The unit wave `t` seconds after the key (0 for `t <= 0`). Multiply by `size` and the travel.
    pub fn at(&self, t: f64) -> f64 {
        let w = self.sanitized();
        if !t.is_finite() || t <= 0.0 {
            return 0.0;
        }
        match w.kind {
            WaveKind::Oscillate => (std::f64::consts::TAU * w.freq * t).sin() * (-w.decay * t).exp(),
            WaveKind::Bounce => {
                let rebound = (-w.decay / 6.0).exp();
                let (mut t, mut d, mut k) = (t, 1.0 / w.freq, 0i32);
                for _ in 0..MAX_BOUNCES {
                    if t <= d {
                        let u = t / d;
                        return -4.0 * u * (1.0 - u) * rebound.powi(2 * k);
                    }
                    t -= d;
                    d *= rebound;
                    k += 1;
                    if d < 0.002 {
                        break;
                    }
                }
                0.0
            }
        }
    }
}

/// Fade factor 0..1 over the 30 % of a segment next to the neighbouring key.
fn fade(seg: f64, from_neighbour: f64) -> f64 {
    if seg > 1e-9 { (from_neighbour / (0.3 * seg)).clamp(0.0, 1.0) } else { 0.0 }
}

fn travel(a: &Value, b: &Value) -> Option<Vec<f64>> {
    let (ca, cb) = (a.components(), b.components());
    (!ca.is_empty() && ca.len() == cb.len()).then(|| ca.iter().zip(&cb).map(|(x, y)| y - x).collect())
}

/// `base` (the tweened value at `t`) plus the waves of the keys around `t`.
pub(crate) fn apply<K: Borrow<Keyframe>>(keys: &[K], t: Tick, base: Value) -> Value {
    if !matches!(base, Value::Scalar(_) | Value::Vec2(_) | Value::Vec3(_)) {
        return base;
    }
    let n = keys.len();
    let key = |i: usize| -> Option<&Keyframe> { keys.get(i).map(|k| k.borrow()) };
    let idx = keys.partition_point(|k| k.borrow().time <= t);
    let prev = idx.checked_sub(1);
    let next = (idx < n).then_some(idx);
    let ts = secs(t);
    let mut off = vec![0.0; base.dims()];
    let mut any = false;

    // After the value arrives at `prev`.
    if let Some(i) = prev.filter(|i| *i >= 1)
        && let (Some(k), Some(before)) = (key(i), key(i - 1))
        && let Some(w) = k.wave_after
        && let Some(d) = travel(&before.value, &k.value)
    {
        let f = match key(i + 1) {
            Some(nk) => fade(secs(nk.time) - secs(k.time), secs(nk.time) - ts),
            None => 1.0,
        };
        let s = w.sanitized().size * w.at(ts - secs(k.time)) * f;
        for (o, dv) in off.iter_mut().zip(&d) {
            *o += dv * s;
        }
        any = true;
    }

    // Before the value leaves `next`.
    if let Some(m) = next
        && let (Some(k), Some(after)) = (key(m), key(m + 1))
        && let Some(w) = k.wave_before
        && let Some(d) = travel(&k.value, &after.value)
    {
        let f = match prev.and_then(key) {
            Some(pk) => fade(secs(k.time) - secs(pk.time), ts - secs(pk.time)),
            None => 1.0,
        };
        let s = -w.sanitized().size * w.at(secs(k.time) - ts) * f;
        for (o, dv) in off.iter_mut().zip(&d) {
            *o += dv * s;
        }
        any = true;
    }

    if !any {
        return base;
    }
    let comps: Vec<f64> = base.components().iter().zip(&off).map(|(b, o)| b + o).collect();
    if comps.iter().any(|c| !c.is_finite()) {
        return base;
    }
    base.with_components(&comps)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{evaluate, time_reverse};

    fn s(x: f64) -> Tick {
        Tick::from_seconds_f64(x)
    }

    fn scalar(v: &Value) -> f64 {
        v.components().first().copied().unwrap_or(f64::NAN)
    }

    /// A 0 → 100 linear tween over 1 s with `wave` after the arrival.
    fn tween(wave: Wave) -> Vec<Keyframe> {
        let a = Keyframe::new(s(0.0), Value::Scalar(0.0));
        let mut b = Keyframe::new(s(1.0), Value::Scalar(100.0));
        b.wave_after = Some(wave);
        vec![a, b]
    }

    #[test]
    fn wave_is_zero_at_the_key_and_decays() {
        for kind in [WaveKind::Oscillate, WaveKind::Bounce] {
            let w = Wave { kind, ..Wave::default() };
            assert_eq!(w.at(0.0), 0.0);
            assert_eq!(w.at(-1.0), 0.0);
            assert!(w.at(30.0).abs() < 1e-6, "{kind:?} should have settled");
        }
    }

    #[test]
    fn oscillate_swings_past_the_end_value_then_settles() {
        let keys = tween(Wave { kind: WaveKind::Oscillate, size: 0.2, freq: 3.0, decay: 4.0 });
        // First quarter period after arrival: the sine is positive, so it overshoots 100.
        let over = evaluate(&keys, s(1.0 + 1.0 / 12.0), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!(over > 100.0 && over < 125.0, "overshoot {over}");
        let end = evaluate(&keys, s(20.0), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!((end - 100.0).abs() < 1e-6, "settled at {end}");
        // The tween itself is untouched.
        let mid = evaluate(&keys, s(0.5), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!((mid - 50.0).abs() < 1e-9, "tween {mid}");
    }

    #[test]
    fn bounce_rebounds_toward_the_start_and_never_passes_the_end() {
        let keys = tween(Wave { kind: WaveKind::Bounce, size: 0.3, freq: 2.0, decay: 6.0 });
        let mut lowest = f64::MAX;
        for i in 0..400 {
            let v = evaluate(&keys, s(1.0 + i as f64 * 0.01), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
            assert!(v <= 100.0 + 1e-9, "bounce passed the end: {v}");
            lowest = lowest.min(v);
        }
        // First rebound peaks at size * travel = 30 below the end.
        assert!((lowest - 70.0).abs() < 1.0, "lowest {lowest}");
    }

    #[test]
    fn wave_before_pulls_back_before_leaving() {
        // Key at 1 s (value 100) leaves toward 200 at 2 s; the wave plays before it leaves.
        let mut a = Keyframe::new(s(1.0), Value::Scalar(100.0));
        a.wave_before = Some(Wave { kind: WaveKind::Oscillate, size: 0.2, freq: 3.0, decay: 2.0 });
        let keys = vec![a, Keyframe::new(s(2.0), Value::Scalar(200.0))];
        // Just before the key it swings against the travel direction (below 100).
        let v = evaluate(&keys, s(0.95), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!(v < 100.0 && v > 70.0, "before the key {v}");
        // At the key and after it the value is the plain tween.
        let at_key = evaluate(&keys, s(1.0), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!((at_key - 100.0).abs() < 1e-9, "at key {at_key}");
        let mid = evaluate(&keys, s(1.5), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!((mid - 150.0).abs() < 1e-9, "tween {mid}");
    }

    #[test]
    fn the_next_tween_starts_exactly_on_its_key() {
        let mut keys = tween(Wave { kind: WaveKind::Oscillate, size: 0.5, freq: 2.0, decay: 0.0 });
        keys.push(Keyframe::new(s(2.0), Value::Scalar(0.0)));
        let at_second = evaluate(&keys, s(2.0), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!((at_second - 0.0).abs() < 1e-9, "next key value {at_second}");
        // Just before it the fade has shrunk the wave to almost nothing.
        let near = evaluate(&keys, s(2.0 - 1e-4), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
        assert!(near.abs() < 1.0, "near next key {near}");
    }

    #[test]
    fn hostile_numbers_never_panic_or_produce_nan() {
        let bad = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0, 1e300, -1e300];
        for &x in &bad {
            for kind in [WaveKind::Oscillate, WaveKind::Bounce] {
                let w = Wave { kind, size: x, freq: x, decay: x };
                let keys = tween(w);
                for t in [0.0, 0.5, 1.0, 1.0001, 3.0, 1e6] {
                    let v = evaluate(&keys, s(t), false).map(|v| scalar(&v)).unwrap_or(f64::NAN);
                    assert!(v.is_finite(), "{w:?} at {t} gave {v}");
                }
            }
        }
    }

    #[test]
    fn colors_and_single_keys_ignore_waves() {
        let mut a = Keyframe::new(s(0.0), Value::Color([0.0, 0.0, 0.0, 1.0]));
        a.wave_after = Some(Wave::default());
        let mut b = Keyframe::new(s(1.0), Value::Color([1.0, 1.0, 1.0, 1.0]));
        b.wave_after = Some(Wave::default());
        let keys = vec![a, b];
        assert_eq!(evaluate(&keys, s(1.2), false), Some(Value::Color([1.0, 1.0, 1.0, 1.0])));
        let one = vec![Keyframe { wave_after: Some(Wave::default()), ..Keyframe::new(s(0.0), Value::Scalar(5.0)) }];
        assert_eq!(evaluate(&one, s(0.4), false), Some(Value::Scalar(5.0)));
    }

    #[test]
    fn time_reverse_swaps_before_and_after() {
        let mut keys = tween(Wave::default());
        assert!(keys[1].wave_after.is_some());
        time_reverse(&mut keys);
        assert!(keys[0].wave_before.is_some() && keys[0].wave_after.is_none());
    }

    #[test]
    fn waves_round_trip_through_serde_and_old_files_still_load() {
        let keys = tween(Wave { kind: WaveKind::Bounce, size: 0.25, freq: 2.5, decay: 7.0 });
        let json = serde_json::to_string(&keys).unwrap_or_default();
        let back: Vec<Keyframe> = serde_json::from_str(&json).unwrap_or_default();
        assert_eq!(back, keys);
        // A key written before waves existed has no wave fields.
        let plain = serde_json::to_string(&Keyframe::new(s(0.0), Value::Scalar(1.0))).unwrap_or_default();
        assert!(!plain.contains("wave"), "{plain}");
        let old: Option<Keyframe> = serde_json::from_str(&plain).ok();
        assert!(old.is_some_and(|k| k.wave_after.is_none() && k.wave_before.is_none()));
    }
}
