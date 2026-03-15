// Copyright (c) 2017 Personal (Binbin Zhang)
// Simplified version for FireRedVAD

#ifndef FRONTEND_FBANK_H_
#define FRONTEND_FBANK_H_

#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#include "frontend/fft.h"

namespace vad {

static const int kS16AbsMax = 1 << 15;

class Fbank {
 public:
  Fbank(int num_bins, int sample_rate, int frame_length, int frame_shift,
        bool use_log = true, bool remove_dc_offset = true,
        bool pre_emphasis = true, float dither = 0.0f)
      : num_bins_(num_bins),
        sample_rate_(sample_rate),
        frame_length_(frame_length),
        frame_shift_(frame_shift),
        use_log_(use_log),
        remove_dc_offset_(remove_dc_offset),
        pre_emphasis_(pre_emphasis),
        dither_(dither),
        pre_emphasis_state_(0.0f) {
    fft_points_ = UpperPowerOfTwo(frame_length_);
    // generate bit reversal table and trigonometric function table
    const int fft_points_4 = fft_points_ / 4;
    bitrev_.resize(fft_points_);
    sintbl_.resize(fft_points_ + fft_points_4);
    make_sintbl(fft_points_, sintbl_.data());
    make_bitrev(fft_points_, bitrev_.data());
    InitMelFilters();
    InitWindow();
  }

  void InitMelFilters() {
    int num_fft_bins = fft_points_ / 2;
    float fft_bin_width = static_cast<float>(sample_rate_) / fft_points_;
    float mel_low_freq = MelScale(20.0f);  // low_freq = 20
    float mel_high_freq = MelScale(sample_rate_ / 2.0f);
    float mel_freq_delta = (mel_high_freq - mel_low_freq) / (num_bins_ + 1);
    bins_.resize(num_bins_);
    center_freqs_.resize(num_bins_);

    for (int bin = 0; bin < num_bins_; ++bin) {
      float left_mel = mel_low_freq + bin * mel_freq_delta;
      float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
      float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;
      center_freqs_[bin] = InverseMelScale(center_mel);
      std::vector<float> this_bin(num_fft_bins);
      int first_index = -1, last_index = -1;
      for (int i = 0; i < num_fft_bins; ++i) {
        float freq = (fft_bin_width * i);
        float mel = MelScale(freq);
        if (mel > left_mel && mel < right_mel) {
          float weight;
          if (mel <= center_mel)
            weight = (mel - left_mel) / (center_mel - left_mel);
          else
            weight = (right_mel - mel) / (right_mel - center_mel);
          this_bin[i] = weight;
          if (first_index == -1) first_index = i;
          last_index = i;
        }
      }
      if (first_index == -1 || last_index < first_index) {
        fprintf(stderr, "Error: invalid mel filter\n");
        continue;
      }
      bins_[bin].first = first_index;
      int size = last_index + 1 - first_index;
      bins_[bin].second.resize(size);
      for (int i = 0; i < size; ++i) {
        bins_[bin].second[i] = this_bin[first_index + i];
      }
    }
  }

  void InitWindow() {
    window_.resize(frame_length_);
    // povey window
    double a = M_2PI / (frame_length_ - 1);
    for (int i = 0; i < frame_length_; ++i)
      window_[i] = pow(0.5 - 0.5 * cos(a * i), 0.85);
  }

  static inline float InverseMelScale(float mel_freq) {
    return 700.0f * (expf(mel_freq / 1127.0f) - 1.0f);
  }

  static inline float MelScale(float freq) {
    return 1127.0f * logf(1.0f + freq / 700.0f);
  }

  static int UpperPowerOfTwo(int n) {
    return static_cast<int>(pow(2, ceil(log(n) / log(2))));
  }

  // pre emphasis
  void PreEmphasis(float coeff, std::vector<float>* data) const {
    if (coeff == 0.0) return;
    for (int i = data->size() - 1; i > 0; i--)
      (*data)[i] -= coeff * (*data)[i - 1];
    (*data)[0] -= coeff * (*data)[0];
  }

  // Apply window on data in place
  void ApplyWindow(std::vector<float>* data) const {
    for (size_t i = 0; i < window_.size(); ++i) {
      (*data)[i] *= window_[i];
    }
  }

  // Compute fbank feat, return num frames
  // Output is flattened: [num_frames * num_bins]
  int Compute(const std::vector<float>& wave, std::vector<float>* feat) {
    int num_samples = wave.size();

    if (num_samples < frame_length_) return 0;
    int num_frames = 1 + ((num_samples - frame_length_) / frame_shift_);
    feat->resize(num_frames * num_bins_);

    std::vector<float> fft_real(fft_points_, 0), fft_img(fft_points_, 0);
    std::vector<float> power(fft_points_ / 2);

    for (int i = 0; i < num_frames; ++i) {
      std::vector<float> data(wave.begin() + i * frame_shift_,
                              wave.begin() + i * frame_shift_ + frame_length_);

      // remove dc offset
      if (remove_dc_offset_) {
        float mean = 0.0;
        for (size_t j = 0; j < data.size(); ++j) mean += data[j];
        mean /= data.size();
        for (size_t j = 0; j < data.size(); ++j) data[j] -= mean;
      }

      // pre emphasis (streaming version, maintains state)
      if (pre_emphasis_) {
        float prev = pre_emphasis_state_;
        for (size_t j = 0; j < data.size(); ++j) {
          float curr = data[j];
          data[j] = curr - 0.97f * prev;
          prev = curr;
        }
        pre_emphasis_state_ = data.back();
      }

      ApplyWindow(&data);

      // FFT
      memset(fft_img.data(), 0, sizeof(float) * fft_points_);
      memset(fft_real.data() + frame_length_, 0,
             sizeof(float) * (fft_points_ - frame_length_));
      memcpy(fft_real.data(), data.data(), sizeof(float) * frame_length_);
      fft(bitrev_.data(), sintbl_.data(), fft_real.data(), fft_img.data(),
          fft_points_);

      // power spectrum
      for (int j = 0; j < fft_points_ / 2; ++j) {
        power[j] = fft_real[j] * fft_real[j] + fft_img[j] * fft_img[j];
      }

      // mel filter bank
      for (int j = 0; j < num_bins_; ++j) {
        float mel_energy = 0.0;
        int s = bins_[j].first;
        for (size_t k = 0; k < bins_[j].second.size(); ++k) {
          mel_energy += bins_[j].second[k] * power[s + k];
        }
        // log
        if (use_log_) {
          if (mel_energy < 1e-20) mel_energy = 1e-20;
          mel_energy = logf(mel_energy);
        }
        (*feat)[i * num_bins_ + j] = mel_energy;
      }
    }

    return num_frames;
  }

  int num_bins() const { return num_bins_; }

  // Reset pre-emphasis state (for new audio)
  void reset() { pre_emphasis_state_ = 0.0f; }

 private:
  int num_bins_;
  int sample_rate_;
  int frame_length_, frame_shift_;
  int fft_points_;
  bool use_log_;
  bool remove_dc_offset_;
  bool pre_emphasis_;
  float dither_;
  float pre_emphasis_state_;

  std::vector<float> center_freqs_;
  std::vector<std::pair<int, std::vector<float>>> bins_;
  std::vector<float> window_;

  // bit reversal table
  std::vector<int> bitrev_;
  // trigonometric function table
  std::vector<float> sintbl_;
};

}  // namespace vad

#endif  // FRONTEND_FBANK_H_
