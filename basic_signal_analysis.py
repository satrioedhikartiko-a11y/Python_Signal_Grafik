import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sg

# ---------------------------------------------------------------
# 1. PARAMETER SINYAL (A = amplitudo, f = frekuensi, phi = fasa, T = periode)
# ---------------------------------------------------------------
A = 1.0          # amplitudo (V)
f = 5.0          # frekuensi (Hz)
phi = 0.0        # fasa (rad)
T = 1.0 / f      # periode (s)
fs = 2000.0      # sampling rate (Hz)
t = np.arange(0, 3 * T, 1 / fs)          # 3 periode untuk sinyal periodik
t_ap = np.linspace(-1.5 * T, 1.5 * T, len(t))  # sumbu waktu utk sinyal step/ramp/impulse

# ---------------------------------------------------------------
# 2. PEMBANGKITAN 8 SINYAL DASAR
# ---------------------------------------------------------------
sine      = A * np.sin(2 * np.pi * f * t + phi)
square    = A * sg.square(2 * np.pi * f * t + phi)
triangle  = A * sg.sawtooth(2 * np.pi * f * t + phi, width=0.5)
sawtooth  = A * sg.sawtooth(2 * np.pi * f * t + phi, width=1.0)
step      = A * (t_ap >= 0).astype(float)
ramp      = A * np.where(t_ap >= 0, t_ap, 0.0)
impulse   = np.zeros_like(t_ap); impulse[np.argmin(np.abs(t_ap))] = A
parabolic = np.where(t_ap >= 0, 0.5 * A * t_ap**2, 0.0)

signals = {
    "Sine":      (t, sine),
    "Square":    (t, square),
    "Triangle":  (t, triangle),
    "Sawtooth":  (t, sawtooth),
    "Unit Step": (t_ap, step),
    "Ramp":      (t_ap, ramp),
    "Impulse":   (t_ap, impulse),
    "Parabolic": (t_ap, parabolic),
}

# ---------------------------------------------------------------
# 3. PLOT DOMAIN WAKTU (8 sinyal)
# ---------------------------------------------------------------
fig, axs = plt.subplots(4, 2, figsize=(11, 12))
for ax, (name, (tt, x)) in zip(axs.flat, signals.items()):
    ax.plot(tt, x, lw=1.5)
    ax.set_title(name)
    ax.set_xlabel("t (s)")
    ax.set_ylabel("Amplitudo")
    ax.grid(alpha=0.3)
fig.suptitle(f"8 Sinyal Dasar (A={A}, f={f} Hz, phi={phi} rad, T={T:.3f} s)", y=1.01)
fig.tight_layout()
fig.savefig("basic_signals_time_domain.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------
# 4. ANALISIS FREKUENSI (FFT) UNTUK SINYAL PERIODIK
# ---------------------------------------------------------------
periodic = {"Sine": sine, "Square": square, "Triangle": triangle, "Sawtooth": sawtooth}
fig2, axs2 = plt.subplots(2, 2, figsize=(11, 7))
for ax, (name, x) in zip(axs2.flat, periodic.items()):
    N = len(x)
    X = np.fft.rfft(x) / N
    freqs = np.fft.rfftfreq(N, d=1 / fs)
    ax.stem(freqs, 2 * np.abs(X))
    ax.set_xlim(0, 10 * f)
    ax.set_title(f"Spektrum Frekuensi — {name}")
    ax.set_xlabel("Frekuensi (Hz)")
    ax.set_ylabel("Magnitudo")
    ax.grid(alpha=0.3)
fig2.tight_layout()
fig2.savefig("basic_signals_frequency_spectrum.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------
# 5. REKONSTRUKSI GELOMBANG KOTAK DARI DERET FOURIER
#    (metode Suparman et al., 2025 — hanya harmonik ganjil)
#    i(wt) = (4A/pi) * sum_{n odd} (1/n) sin(n*w*t)
# ---------------------------------------------------------------
def square_fourier_reconstruct(t, A, f, n_terms):
    w = 2 * np.pi * f
    x = np.zeros_like(t)
    for n in range(1, 2 * n_terms, 2):  # 1, 3, 5, 7, ...
        x += (4 * A / (np.pi * n)) * np.sin(n * w * t)
    return x

fig3, axs3 = plt.subplots(1, 3, figsize=(14, 4), sharey=True)
for ax, N in zip(axs3, [3, 9, 23]):
    recon = square_fourier_reconstruct(t, A, f, N)
    ax.plot(t, square, "--", label="Ideal", alpha=0.6)
    ax.plot(t, recon, label=f"Rekonstruksi ({N} harmonik)")
    ax.set_title(f"N = {N} harmonik ganjil")
    ax.set_xlabel("t (s)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
axs3[0].set_ylabel("Amplitudo")
fig3.suptitle("Gelombang Kotak: Ideal vs Rekonstruksi Deret Fourier", y=1.03)
fig3.tight_layout()
fig3.savefig("square_wave_ideal_vs_reconstructed.png", dpi=150, bbox_inches="tight")

# ---------------------------------------------------------------
# 6. HARMONIC CONTENT: Crest Factor (CF) & Total Harmonic Distortion (THD)
#    Mengikuti persamaan (6)-(10) pada Suparman et al. (2025)
# ---------------------------------------------------------------
n_harmonics = np.arange(1, 24, 2)                 # 1,3,5,...,23 (ganjil)
peak = (4 * A) / (np.pi * n_harmonics)             # amplitudo tiap harmonik
rms = peak / np.sqrt(2)

fundamental = peak[0]
harmonics_only = peak[1:]                          # H3, H5, ..., H23
THDF = np.sqrt(np.sum(harmonics_only**2)) / fundamental
THDR = THDF / np.sqrt(1 + THDF**2)

peak_val = np.max(np.abs(square))
rms_val = np.sqrt(np.mean(square**2))
CF = peak_val / rms_val

print("=== Analisis Harmonik Gelombang Kotak (mengikuti Suparman et al., 2025) ===")
print(f"Crest Factor (CF)              : {CF:.3f}")
print(f"Total Harmonic Distortion THDF : {THDF*100:.1f} %")
print(f"Total Harmonic Distortion THDR : {THDR*100:.1f} %")
print("\nTabel amplitudo harmonik (n, peak, RMS):")
for n, p, r in zip(n_harmonics, peak, rms):
    print(f"  n={n:2d}  peak={p:.3f}  rms={r:.3f}")

# ---------------------------------------------------------------
# 7. RINGKASAN PERIODISITAS
# ---------------------------------------------------------------
print("\n=== Periodisitas Sinyal ===")
for name in ["Sine", "Square", "Triangle", "Sawtooth"]:
    print(f"{name:10s}: periodik, T = {T:.4f} s, f = {f} Hz")
for name in ["Unit Step", "Ramp", "Impulse", "Parabolic"]:
    print(f"{name:10s}: aperiodik (non-periodic)")

plt.show()
print("\nSelesai. Grafik disimpan sebagai:")
print(" - basic_signals_time_domain.png")
print(" - basic_signals_frequency_spectrum.png")
print(" - square_wave_ideal_vs_reconstructed.png")
