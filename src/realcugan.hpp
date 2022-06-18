// realcugan implemented with ncnn library

#ifndef REALCUGAN_HPP
#define REALCUGAN_HPP

#include <string>

// ncnn
#include "net.h"
#include "gpu.h"
#include "layer.h"

constexpr int CHANNELS = 3;

class FeatureCache;
class RealCUGAN
{
public:
    RealCUGAN(int gpuid, bool _tta_mode);
    ~RealCUGAN();

    int load(const std::string& paramPath, const std::string& modelpath);
    
    int process(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;

    int process_cpu(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;
    
    int process_se(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;
    
    int process_cpu_se(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;
    
    int process_se_rough(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;
    
    int process_cpu_se_rough(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;
    
    int process_se_very_rough(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;
    
    int process_cpu_se_very_rough(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const;

protected:
    int process_se_stage0(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const std::vector<std::string>& outnames, const ncnn::Option& opt, FeatureCache& cache) const;
    int process_se_stage2(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache) const;
    int process_se_sync_gap(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache) const;

    int process_se_very_rough_stage0(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const std::vector<std::string>& outnames, const ncnn::Option& opt, FeatureCache& cache) const;
    int process_se_very_rough_sync_gap(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache) const;

    int process_cpu_se_stage0(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const std::vector<std::string>& outnames, FeatureCache& cache) const;
    int process_cpu_se_stage2(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride, const std::vector<std::string>& names, FeatureCache& cache) const;
    int process_cpu_se_sync_gap(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, FeatureCache& cache) const;

    int process_cpu_se_very_rough_stage0(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const std::vector<std::string>& outnames, FeatureCache& cache) const;
    int process_cpu_se_very_rough_sync_gap(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, FeatureCache& cache) const;

public:
    // realcugan parameters
    int noise;
    int scale;
    int tilesize_x;
    int tilesize_y;
    int prepadding;
    int syncgap;

private:
    ncnn::VulkanDevice* vkdev;
    ncnn::Net net;
    ncnn::Pipeline* realcugan_preproc;
    ncnn::Pipeline* realcugan_postproc;
    ncnn::Pipeline* realcugan_4x_postproc;
    bool tta_mode;
};

#endif  // REALCUGAN_HPP
