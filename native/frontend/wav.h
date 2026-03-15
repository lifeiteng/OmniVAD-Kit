// Copyright (c) 2016 Personal (Binbin Zhang)
// Simplified version for FireRedVAD

#ifndef FRONTEND_WAV_H_
#define FRONTEND_WAV_H_

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

namespace vad {

struct WavHeader {
  char riff[4];
  unsigned int size;
  char wav[4];
  char fmt[4];
  unsigned int fmt_size;
  uint16_t format;
  uint16_t channels;
  unsigned int sample_rate;
  unsigned int bytes_per_second;
  uint16_t block_size;
  uint16_t bit;
  char data[4];
  unsigned int data_size;
};

class WavReader {
 public:
  WavReader() : data_(nullptr) {}
  explicit WavReader(const std::string& filename) { Open(filename); }

  bool Open(const std::string& filename) {
    FILE* fp = fopen(filename.c_str(), "rb");
    if (NULL == fp) {
      fprintf(stderr, "Error opening %s\n", filename.c_str());
      return false;
    }

    WavHeader header;
    fread(&header, 1, sizeof(header), fp);

    // Check RIFF, WAVE, fmt
    if ((0 != strncmp(header.riff, "RIFF", 4)) ||
        (0 != strncmp(header.wav, "WAVE", 4)) ||
        (0 != strncmp(header.fmt, "fmt", 3))) {
      fprintf(stderr, "Invalid WAV file format\n");
      fclose(fp);
      return false;
    }

    // Skip any sub-chunks between "fmt" and "data"
    while (0 != strncmp(header.data, "data", 4)) {
      // skip current chunk payload
      if (fseek(fp, header.data_size, SEEK_CUR) != 0) {
        fprintf(stderr, "Failed to seek over chunk\n");
        fclose(fp);
        return false;
      }
      // read next chunk id (4 bytes)
      if (fread(header.data, 1, 4, fp) != 4) {
        fprintf(stderr, "Failed to find data chunk\n");
        fclose(fp);
        return false;
      }
      // read next chunk size (4 bytes)
      if (fread(&header.data_size, 1, 4, fp) != 4) {
        fprintf(stderr, "Failed to read next chunk size\n");
        fclose(fp);
        return false;
      }
    }

    num_channel_ = header.channels;
    sample_rate_ = header.sample_rate;
    bits_per_sample_ = header.bit;

    if (sample_rate_ != 16000) {
      fprintf(stderr, "Warning: sample rate is %d, expected 16000\n", sample_rate_);
    }

    int num_data = header.data_size / (bits_per_sample_ / 8);
    data_ = new float[num_data];
    num_samples_ = num_data / num_channel_;

    for (int i = 0; i < num_data; ++i) {
      switch (bits_per_sample_) {
        case 8: {
          char sample;
          fread(&sample, 1, sizeof(char), fp);
          data_[i] = static_cast<float>(sample);
          break;
        }
        case 16: {
          int16_t sample;
          fread(&sample, 1, sizeof(int16_t), fp);
          data_[i] = static_cast<float>(sample);
          break;
        }
        case 32: {
          int sample;
          fread(&sample, 1, sizeof(int), fp);
          data_[i] = static_cast<float>(sample);
          break;
        }
        default:
          fprintf(stderr, "unsupported quantization bits: %d\n", bits_per_sample_);
          delete[] data_;
          data_ = nullptr;
          fclose(fp);
          return false;
      }
    }
    fclose(fp);
    return true;
  }

  int num_channel() const { return num_channel_; }
  int sample_rate() const { return sample_rate_; }
  int bits_per_sample() const { return bits_per_sample_; }
  int num_samples() const { return num_samples_; }

  ~WavReader() { delete[] data_; }

  const float* data() const { return data_; }

  // Get mono data (average channels if stereo)
  std::vector<float> GetMonoData() const {
    std::vector<float> mono(num_samples_);
    if (num_channel_ == 1) {
      for (int i = 0; i < num_samples_; i++) {
        mono[i] = data_[i];
      }
    } else {
      for (int i = 0; i < num_samples_; i++) {
        float sum = 0;
        for (int c = 0; c < num_channel_; c++) {
          sum += data_[i * num_channel_ + c];
        }
        mono[i] = sum / num_channel_;
      }
    }
    return mono;
  }

 private:
  int num_channel_;
  int sample_rate_;
  int bits_per_sample_;
  int num_samples_;
  float* data_;
};

}  // namespace vad

#endif  // FRONTEND_WAV_H_
