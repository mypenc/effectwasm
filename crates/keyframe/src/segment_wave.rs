//! A bounce / elastic / spring / oscillate curve that *is* the motion between two keyframes.
//!
//! Stored on the key a segment starts at (`Keyframe::segment_wave`). Inside the segment the value
//! is `a + (b - a) * f(x)` with `x` the 0..1 progress, `f(0) = 0` and `f(1) = 1`: it swings or
//! bounces while it travels and lands exactly on the second key when the segment ends, as
//! Alight Motion's bounce / oscillate easings do. Nothing plays outside the segment (see
//! [`crate::wave`] for waves that play after or before a key instead).
//!
//! Only scalar, 2D and 3D values take one. On a spatial property the value moves along the
//! straight line between the keys.

use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentKind {
    /// A ball dropped onto the end value: each rebound is lower (`count` bounces).
    #[default]
    Bounce,
    /// An elastic snap: overshoots and rings around the end value (`count` swings).
    Elastic,
    /// A damped spring: like elastic with the decay tied to `size`.
    Spring,
    /// An ease with a damped sine laid on it (`count` swings, `decay` how fast it dies out).
    Oscillate,
}

impl SegmentKind {
    pub fn label(self) -> &'static str {
        match self {
            SegmentKind::Bounce => "Bounce",
            SegmentKind::Elastic => "Elastic",
            SegmentKind::Spring => "Spring",
            SegmentKind::Oscillate => "Oscillate",
        }
    }
}

/// Which end of the segment the wave plays at.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SegmentDir {
    /// Settles onto the second key.
    #[default]
    Out,
    /// Builds up from the first key.
    In,
    /// Both ends.
    InOut,
}

impl SegmentDir {
    /// The same curve run backwards through the segment (point reflection).
    pub fn mirrored(self) -> SegmentDir {
        match self {
            SegmentDir::Out => SegmentDir::In,
            SegmentDir::In => SegmentDir::Out,
            SegmentDir::InOut => SegmentDir::InOut,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct SegmentWave {
    pub kind: SegmentKind,
    pub dir: SegmentDir,
    /// Bounces / swings inside the segment (1..=16).
    pub count: u32,
    /// Wave size 0..1: how high the bounces are, how hard the swings overshoot.
    pub size: f64,
    /// Oscillate only: how fast the swings die out.
    pub decay: f64,
}

impl Default for SegmentWave {
    fn default() -> Self {
        SegmentWave { kind: SegmentKind::Bounce, dir: SegmentDir::Out, count: 3, size: 0.55, decay: 4.0 }
    }
}

fn smooth(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

impl SegmentWave {
    /// A copy with every number finite and inside the accepted range.
    pub fn sanitized(self) -> SegmentWave {
        let fix = |v: f64, lo: f64, hi: f64, or: f64| if v.is_finite() { v.clamp(lo, hi) } else { or };
        let d = SegmentWave::default();
        SegmentWave {
            kind: self.kind,
            dir: self.dir,
            count: self.count.clamp(1, 16),
            size: fix(self.size, 0.0, 1.0, d.size),
            decay: fix(self.decay, 0.0, 30.0, d.decay),
        }
    }

    /// The same shape run backwards (used when the keys are time-reversed).
    pub fn mirrored(self) -> SegmentWave {
        SegmentWave { dir: self.dir.mirrored(), ..self }
    }

    /// The "Out" form: `f(0) = 0`, `f(1) = 1`.
    fn out(&self, x: f64) -> f64 {
        let n = f64::from(self.count);
        let tau = std::f64::consts::TAU;
        // Settle exactly onto the end value over the last 15 % for the ringing shapes.
        let settle = |f: f64| f + (1.0 - f) * smooth((x - 0.85) / 0.15);
        match self.kind {
            SegmentKind::Bounce => {
                let e = 0.15 + 0.75 * self.size;
                let total = 1.0 + (1..=self.count).map(|k| 2.0 * e.powi(k as i32)).sum::<f64>();
                let mut s = x * total;
                if s <= 1.0 {
                    return s * s;
                }
                s -= 1.0;
                for k in 1..=self.count {
                    let c = e.powi(k as i32);
                    let d = 2.0 * c;
                    if s <= d {
                        let u = s - c;
                        return 1.0 - c * c + u * u;
                    }
                    s -= d;
                }
                1.0
            }
            SegmentKind::Elastic => {
                let decay = 2.0 + 10.0 * (1.0 - self.size);
                settle(1.0 - (-decay * x).exp() * (tau * n * x).cos())
            }
            SegmentKind::Spring => {
                let z = 0.9 - 0.8 * self.size;
                let wd = tau * n;
                let w0 = wd / (1.0 - z * z).sqrt();
                settle(1.0 - (-z * w0 * x).exp() * ((wd * x).cos() + (z * w0 / wd) * (wd * x).sin()))
            }
            SegmentKind::Oscillate => smooth(x) + 0.5 * self.size * (tau * n * x).sin() * (-self.decay * x).exp(),
        }
    }

    /// Progress through the segment for `x` in 0..1 (`f(0) = 0`, `f(1) = 1`; may overshoot).
    pub fn at(&self, x: f64) -> f64 {
        let w = self.sanitized();
        if !x.is_finite() || x <= 0.0 {
            return 0.0;
        }
        if x >= 1.0 {
            return 1.0;
        }
        let v = match w.dir {
            SegmentDir::Out => w.out(x),
            SegmentDir::In => 1.0 - w.out(1.0 - x),
            SegmentDir::InOut => {
                if x < 0.5 {
                    0.5 * (1.0 - w.out(1.0 - 2.0 * x))
                } else {
                    0.5 + 0.5 * w.out(2.0 * x - 1.0)
                }
            }
        };
        if v.is_finite() { v } else { x }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Keyframe, Tick, Value, evaluate, time_reverse};

    const KINDS: [SegmentKind; 4] = [SegmentKind::Bounce, SegmentKind::Elastic, SegmentKind::Spring, SegmentKind::Oscillate];
    const DIRS: [SegmentDir; 3] = [SegmentDir::Out, SegmentDir::In, SegmentDir::InOut];

    fn s(x: f64) -> Tick {
        Tick::from_seconds_f64(x)
    }

    fn scalar(v: Option<Value>) -> f64 {
        v.and_then(|v| v.components().first().copied()).unwrap_or(f64::NAN)
    }

    /// 0 → 100 over one second with the wave on the first key.
    fn seg(w: SegmentWave) -> Vec<Keyframe> {
        let mut a = Keyframe::new(s(0.0), Value::Scalar(0.0));
        a.segment_wave = Some(w);
        vec![a, Keyframe::new(s(1.0), Value::Scalar(100.0))]
    }

    #[test]
    fn every_shape_starts_at_zero_and_lands_exactly_on_one() {
        for kind in KINDS {
            for dir in DIRS {
                for count in [1, 3, 8, 16] {
                    for size in [0.0, 0.5, 1.0] {
                        let w = SegmentWave { kind, dir, count, size, decay: 4.0 };
                        assert_eq!(w.at(0.0), 0.0, "{w:?}");
                        assert_eq!(w.at(1.0), 1.0, "{w:?}");
                        assert!((w.at(1.0 - 1e-9) - 1.0).abs() < 1e-3, "{w:?} must arrive on 1, got {}", w.at(1.0 - 1e-9));
                        for i in 0..=200 {
                            assert!(w.at(f64::from(i) / 200.0).is_finite(), "{w:?}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn the_motion_swings_inside_the_segment_and_ends_when_it_ends() {
        // Bottom (0) to top (100): an elastic swings past the top and back before the end key.
        let keys = seg(SegmentWave { kind: SegmentKind::Elastic, dir: SegmentDir::Out, count: 3, size: 0.6, decay: 4.0 });
        let samples: Vec<f64> = (0..=100).map(|i| scalar(evaluate(&keys, s(f64::from(i) / 100.0), false))).collect();
        assert!(samples.iter().any(|v| *v > 100.5), "it overshoots the top");
        assert!(samples.windows(3).filter(|w| (w[1] - w[0]) * (w[2] - w[1]) < 0.0).count() >= 2, "it goes up and down more than once");
        assert!((samples[100] - 100.0).abs() < 1e-9, "it ends on the second key");
        // Nothing plays after the end key.
        assert!((scalar(evaluate(&keys, s(1.5), false)) - 100.0).abs() < 1e-9);
        assert!((scalar(evaluate(&keys, s(30.0), false)) - 100.0).abs() < 1e-9);
    }

    #[test]
    fn bounce_out_drops_onto_the_end_and_never_passes_it() {
        let keys = seg(SegmentWave { kind: SegmentKind::Bounce, dir: SegmentDir::Out, count: 4, size: 0.6, decay: 0.0 });
        let mut touches = 0;
        let mut prev = 0.0;
        for i in 0..=1000 {
            let v = scalar(evaluate(&keys, s(f64::from(i) / 1000.0), false));
            assert!(v <= 100.0 + 1e-9, "passed the end: {v}");
            if i > 0 && prev < 99.0 && v >= 99.0 {
                touches += 1;
            }
            prev = v;
        }
        assert!(touches >= 4, "touched the floor {touches} times");
    }

    #[test]
    fn works_on_2d_values_and_leaves_other_kinds_alone() {
        let mut a = Keyframe::new(s(0.0), Value::Vec2([0.0, 1000.0]));
        a.segment_wave = Some(SegmentWave::default());
        let keys = vec![a, Keyframe::new(s(1.0), Value::Vec2([0.0, 0.0]))];
        let end = evaluate(&keys, s(1.0), true).map(|v| v.components()).unwrap_or_default();
        assert_eq!(end, vec![0.0, 0.0]);
        let mut c = Keyframe::new(s(0.0), Value::Color([0.0, 0.0, 0.0, 1.0]));
        c.segment_wave = Some(SegmentWave::default());
        let keys = vec![c, Keyframe::new(s(1.0), Value::Color([1.0, 1.0, 1.0, 1.0]))];
        assert_eq!(evaluate(&keys, s(0.5), false).map(|v| v.components().first().copied()), Some(Some(0.5)));
    }

    #[test]
    fn hostile_numbers_never_panic() {
        for x in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 0.0, 1e300] {
            for kind in KINDS {
                let w = SegmentWave { kind, dir: SegmentDir::InOut, count: u32::MAX, size: x, decay: x };
                for i in 0..=50 {
                    assert!(w.at(f64::from(i) / 50.0).is_finite(), "{w:?}");
                }
                assert!(w.at(x).is_finite());
            }
        }
    }

    #[test]
    fn time_reverse_moves_the_curve_to_the_new_start_and_flips_it() {
        let mut keys = seg(SegmentWave { dir: SegmentDir::Out, ..SegmentWave::default() });
        keys.push(Keyframe::new(s(2.0), Value::Scalar(0.0)));
        time_reverse(&mut keys);
        // The old first segment is now the last one, run backwards, so it starts at keys[1].
        assert!(keys[0].segment_wave.is_none() && keys[2].segment_wave.is_none());
        assert_eq!(keys[1].segment_wave.map(|w| w.dir), Some(SegmentDir::In));
    }

    #[test]
    fn round_trips_through_serde_and_is_omitted_when_unset() {
        let keys = seg(SegmentWave { kind: SegmentKind::Spring, dir: SegmentDir::InOut, count: 5, size: 0.7, decay: 2.0 });
        let json = serde_json::to_string(&keys).unwrap_or_default();
        let back: Vec<Keyframe> = serde_json::from_str(&json).unwrap_or_default();
        assert_eq!(back, keys);
        let plain = serde_json::to_string(&Keyframe::new(s(0.0), Value::Scalar(1.0))).unwrap_or_default();
        assert!(!plain.contains("segment_wave"), "{plain}");
    }
}
