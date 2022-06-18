// realcugan implemented with ncnn library

#include "realcugan.hpp"

#include <algorithm>
#include <vector>
#include <map>

// ncnn
#include "cpu.h"

#include "realcugan_preproc.comp.hex.h"
#include "realcugan_postproc.comp.hex.h"
#include "realcugan_4x_postproc.comp.hex.h"
#include "realcugan_preproc_tta.comp.hex.h"
#include "realcugan_postproc_tta.comp.hex.h"
#include "realcugan_4x_postproc_tta.comp.hex.h"

class FeatureCache
{
public:
    void clear()
    {
        gpu_cache.clear();
        cpu_cache.clear();
    }

    std::string make_key(int yi, int xi, int ti, const std::string& name) const
    {
        return std::to_string(yi) + "-" + std::to_string(xi) + "-" + std::to_string(ti) + "-" + name;
    }

    void load(int yi, int xi, int ti, const std::string& name, ncnn::VkMat& feat)
    {
        feat = gpu_cache[make_key(yi, xi, ti, name)];
    }

    void save(int yi, int xi, int ti, const std::string& name, ncnn::VkMat& feat)
    {
        gpu_cache[make_key(yi, xi, ti, name)] = feat;
    }

    void load(int yi, int xi, int ti, const std::string& name, ncnn::Mat& feat)
    {
        feat = cpu_cache[make_key(yi, xi, ti, name)];
    }

    void save(int yi, int xi, int ti, const std::string& name, ncnn::Mat& feat)
    {
        cpu_cache[make_key(yi, xi, ti, name)] = feat;
    }

public:
    std::map<std::string, ncnn::VkMat> gpu_cache;
    std::map<std::string, ncnn::Mat> cpu_cache;
};

RealCUGAN::RealCUGAN(int gpuid, bool _tta_mode)
{
    vkdev = gpuid == -1 ? 0 : ncnn::get_gpu_device(gpuid);

    realcugan_preproc = 0;
    realcugan_postproc = 0;
    realcugan_4x_postproc = 0;
    tta_mode = _tta_mode;
}

RealCUGAN::~RealCUGAN()
{
    // clean up preprocess and postprocess pipeline
    delete realcugan_preproc;
    delete realcugan_postproc;
    delete realcugan_4x_postproc;
}

int RealCUGAN::load(const std::string& parampath, const std::string& modelpath)
{
    net.opt.use_vulkan_compute = true;
    net.opt.use_fp16_packed = true;
    net.opt.use_fp16_storage = true;
    net.opt.use_fp16_arithmetic = false;
    net.opt.use_int8_storage = false;  // true

    net.set_vulkan_device(vkdev);

    net.load_param(parampath.c_str());
    net.load_model(modelpath.c_str());

    // initialize preprocess and postprocess pipeline
    if (vkdev)
    {
        std::vector<ncnn::vk_specialization_type> specializations(1);
#if _WIN32
        specializations[0].i = 1;
#else
        specializations[0].i = 0;
#endif

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;

            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                if (tta_mode)
                {
                    compile_spirv_module(realcugan_preproc_tta_comp_data, sizeof(realcugan_preproc_tta_comp_data), net.opt, spirv);
                }
                else
                {
                    compile_spirv_module(realcugan_preproc_comp_data, sizeof(realcugan_preproc_comp_data), net.opt, spirv);
                }

            }

            realcugan_preproc = new ncnn::Pipeline(vkdev);
            realcugan_preproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_preproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;

            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                if (tta_mode)
                {
                    compile_spirv_module(realcugan_postproc_tta_comp_data, sizeof(realcugan_postproc_tta_comp_data), net.opt, spirv);
                }
                else
                {
                    compile_spirv_module(realcugan_postproc_comp_data, sizeof(realcugan_postproc_comp_data), net.opt, spirv);
                }
            }

            realcugan_postproc = new ncnn::Pipeline(vkdev);
            realcugan_postproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }

        {
            static std::vector<uint32_t> spirv;
            static ncnn::Mutex lock;
            
            ncnn::MutexLockGuard guard(lock);
            if (spirv.empty())
            {
                if (tta_mode)
                {
                    compile_spirv_module(realcugan_4x_postproc_tta_comp_data, sizeof(realcugan_4x_postproc_tta_comp_data), net.opt, spirv);
                }
                else
                {
                    compile_spirv_module(realcugan_4x_postproc_comp_data, sizeof(realcugan_4x_postproc_comp_data), net.opt, spirv);
                }
            }

            realcugan_4x_postproc = new ncnn::Pipeline(vkdev);
            realcugan_4x_postproc->set_optimal_local_size_xyz(8, 8, 3);
            realcugan_4x_postproc->create(spirv.data(), spirv.size() * 4, specializations);
        }
    }

    return 0;
}

int RealCUGAN::process(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const
{
    bool syncgap_needed = tilesize_x < width || tilesize_y < height;

    /*
    // cpu only
    if (syncgap_needed && syncgap)
    {
        if (syncgap == 1)
        {
            return process_cpu_se(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride);
        }
        if (syncgap == 2)
        {
            return process_cpu_se_rough(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride);
        }
        if (syncgap == 3)
        {
            return process_cpu_se_very_rough(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride);
        }
    }
    else
    {
        return proceess_cpu(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride);
    }
    */

    if (syncgap_needed && syncgap)
    {
        if (syncgap == 1)
        {
            return process_se(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride);
        }
        if (syncgap == 2)
        {
            return process_se_rough(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride);
        }
        if (syncgap == 3)
        {
            return process_se_very_rough(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride);
        }
    }
    
    const int TILE_SIZE_X = tilesize_x;
    const int TILE_SIZE_Y = tilesize_y;

    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    // each tile 400x400
    const int xtiles = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, height) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, height);
        const int in_tile_w = width;
        const int in_tile_h = in_tile_y1 - in_tile_y0;

        ncnn::Mat in(in_tile_w, in_tile_h, CHANNELS, sizeof(float));

        {
            float* in_tile_r = in.channel(0);
            float* in_tile_g = in.channel(1);
            float* in_tile_b = in.channel(2);
        
            const float* sr = srcpR + in_tile_y0 * src_stride;
            const float* sg = srcpG + in_tile_y0 * src_stride;
            const float* sb = srcpB + in_tile_y0 * src_stride;
        
            for (int y = 0; y < in_tile_h; y++)
            {
                for (int x = 0; x < in_tile_w; x++)
                {
                    in_tile_r[in_tile_w * y + x] = sr[src_stride * y + x] * 255.f;
                    in_tile_g[in_tile_w * y + x] = sg[src_stride * y + x] * 255.f;
                    in_tile_b[in_tile_w * y + x] = sb[src_stride * y + x] * 255.f;
                }
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height);

        ncnn::VkMat out_gpu;
        out_gpu.create(width * scale, (out_tile_y1 - out_tile_y0) * scale, CHANNELS, sizeof(float), blob_vkallocator);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, width) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }
            
            if (tta_mode)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, 3, in_out_tile_elemsize, 1, blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    ex.extract("out0", out_tile_gpu[ti], cmd);
                }

                // postproc
                if (scale == 4)
                {
                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu[0];
                    bindings[2] = out_tile_gpu[1];
                    bindings[3] = out_tile_gpu[2];
                    bindings[4] = out_tile_gpu[3];
                    bindings[5] = out_tile_gpu[4];
                    bindings[6] = out_tile_gpu[5];
                    bindings[7] = out_tile_gpu[6];
                    bindings[8] = out_tile_gpu[7];
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(14);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu[0].w;
                    constants[4].i = out_tile_gpu[0].h;
                    constants[5].i = out_tile_gpu[0].cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(9);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(blob_vkallocator);
                    ex.set_workspace_vkallocator(blob_vkallocator);
                    ex.set_staging_vkallocator(staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    ex.extract("out0", out_tile_gpu, cmd);
                }

                // postproc
                if (scale == 4)
                {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(14);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu.w;
                    constants[4].i = out_tile_gpu.h;
                    constants[5].i = out_tile_gpu.cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(9);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // download
        {
            ncnn::Mat out;

            cmd.record_clone(out_gpu, out, opt);
            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage))
            {
                const float* out_tile_r = out.channel(0);
                const float* out_tile_g = out.channel(1);
                const float* out_tile_b = out.channel(2);

                float* dr = dstpR + yi * TILE_SIZE_Y * scale * dst_stride;
                float* dg = dstpG + yi * TILE_SIZE_Y * scale * dst_stride;
                float* db = dstpB + yi * TILE_SIZE_Y * scale * dst_stride;

                for (int y = 0; y < out.h; y++)
                {
                    for (int x = 0; x < out.w; x++)
                    {
                        dr[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_r[out.w * y + x] / 255.f));
                        dg[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_g[out.w * y + x] / 255.f));
                        db[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_b[out.w * y + x] / 255.f));
                    }
                }
            }
        }
    }

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}


int RealCUGAN::process_se(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const
{
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0"};
    process_se_stage0(srcpR, srcpG, srcpB, width, height, src_stride, in0, out0, opt, cache);

    std::vector<std::string> gap0 = {"gap0"};
    process_se_sync_gap(srcpR, srcpG, srcpB, width, height, src_stride, gap0, opt, cache);

    std::vector<std::string> in1 = {"gap0"};
    std::vector<std::string> out1 = {"gap1"};
    process_se_stage0(srcpR, srcpG, srcpB, width, height, src_stride, in1, out1, opt, cache);

    std::vector<std::string> gap1 = {"gap1"};
    process_se_sync_gap(srcpR, srcpG, srcpB, width, height, src_stride, gap1, opt, cache);

    std::vector<std::string> in2 = {"gap0", "gap1"};
    std::vector<std::string> out2 = {"gap2"};
    process_se_stage0(srcpR, srcpG, srcpB, width, height, src_stride, in2, out2, opt, cache);

    std::vector<std::string> gap2 = {"gap2"};
    process_se_sync_gap(srcpR, srcpG, srcpB, width, height, src_stride, gap2, opt, cache);

    std::vector<std::string> in3 = {"gap0", "gap1", "gap2"};
    std::vector<std::string> out3 = {"gap3"};
    process_se_stage0(srcpR, srcpG, srcpB, width, height, src_stride, in3, out3, opt, cache);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage2(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride, in4, opt, cache);

    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RealCUGAN::process_se_rough(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const
{
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage0(srcpR, srcpG, srcpB, width, height, src_stride, in0, out0, opt, cache);

    std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_sync_gap(srcpR, srcpG, srcpB, width, height, src_stride, gap0, opt, cache);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage2(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride, in4, opt, cache);

    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RealCUGAN::process_se_very_rough(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride) const
{
    ncnn::VkAllocator* blob_vkallocator = vkdev->acquire_blob_allocator();
    ncnn::VkAllocator* staging_vkallocator = vkdev->acquire_staging_allocator();

    ncnn::Option opt = net.opt;
    opt.blob_vkallocator = blob_vkallocator;
    opt.workspace_vkallocator = blob_vkallocator;
    opt.staging_vkallocator = staging_vkallocator;

    FeatureCache cache;

    std::vector<std::string> in0 = {};
    std::vector<std::string> out0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_very_rough_stage0(srcpR, srcpG, srcpB, width, height, src_stride, in0, out0, opt, cache);

    std::vector<std::string> gap0 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_very_rough_sync_gap(srcpR, srcpG, srcpB, width, height, src_stride, gap0, opt, cache);

    std::vector<std::string> in4 = {"gap0", "gap1", "gap2", "gap3"};
    process_se_stage2(srcpR, srcpG, srcpB, dstpR, dstpG, dstpB, width, height, src_stride, dst_stride, in4, opt, cache);

    cache.clear();

    vkdev->reclaim_blob_allocator(blob_vkallocator);
    vkdev->reclaim_staging_allocator(staging_vkallocator);

    return 0;
}

int RealCUGAN::process_se_stage0(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const std::vector<std::string>& outnames, const ncnn::Option& opt, FeatureCache& cache) const
{
    const int TILE_SIZE_X = tilesize_x;
    const int TILE_SIZE_Y = tilesize_y;

    // each tile 400x400 
    const int xtiles = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, height) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, height);
        const int in_tile_w = width;
        const int in_tile_h = in_tile_y1 - in_tile_y0;

        ncnn::Mat in(in_tile_w, in_tile_h, CHANNELS, sizeof(float));

        float* in_tile_r = in.channel(0);
        float* in_tile_g = in.channel(1);
        float* in_tile_b = in.channel(2);

        const float* sr = srcpR + in_tile_y0 * src_stride;
        const float* sg = srcpG + in_tile_y0 * src_stride;
        const float* sb = srcpB + in_tile_y0 * src_stride;

        for (int y = 0; y < in_tile_h; y++)
        {
            for (int x = 0; x < in_tile_w; x++)
            {
                in_tile_r[in_tile_w * y + x] = sr[src_stride * y + x] * 255.f;
                in_tile_g[in_tile_w * y + x] = sg[src_stride * y + x] * 255.f;
                in_tile_b[in_tile_w * y + x] = sb[src_stride * y + x] * 255.f;
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height);

        ncnn::VkMat out_gpu;
        out_gpu.create(width * scale, (out_tile_y1 - out_tile_y0) * scale, CHANNELS, sizeof(float), opt.blob_vkallocator);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, width) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_mode)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];
                
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;
                    
                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        cmd.submit_and_wait();
        cmd.reset();
    }

    return 0;
}

int RealCUGAN::process_se_stage2(const float* srcpR, const float* srcpG, const float* srcpB, float* dstpR, float* dstpG, float* dstpB, int width, int height, int src_stride, int dst_stride, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache) const
{
    const int TILE_SIZE_X = tilesize_x;
    const int TILE_SIZE_Y = tilesize_y;

    // each tile 400x400 
    const int xtiles = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi < ytiles; yi++)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, height) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, height);
        const int in_tile_w = width;
        const int in_tile_h = in_tile_y1 - in_tile_y0;

        ncnn::Mat in(in_tile_w, in_tile_h, CHANNELS, sizeof(float));

        float* in_tile_r = in.channel(0);
        float* in_tile_g = in.channel(1);
        float* in_tile_b = in.channel(2);

        const float* sr = srcpR + in_tile_y0 * src_stride;
        const float* sg = srcpG + in_tile_y0 * src_stride;
        const float* sb = srcpB + in_tile_y0 * src_stride;

        for (int y = 0; y < in_tile_h; y++)
        {
            for (int x = 0; x < in_tile_w; x++)
            {
                in_tile_r[in_tile_w * y + x] = sr[src_stride * y + x] * 255.f;
                in_tile_g[in_tile_w * y + x] = sg[src_stride * y + x] * 255.f;
                in_tile_b[in_tile_w * y + x] = sb[src_stride * y + x] * 255.f;
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height);

        ncnn::VkMat out_gpu;
        out_gpu.create(width * scale, (out_tile_y1 - out_tile_y0) * scale, CHANNELS, sizeof(float), opt.blob_vkallocator);

        for (int xi = 0; xi < xtiles; xi++)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, width) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_mode)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];

                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    ex.extract("out0", out_tile_gpu[ti], cmd);
                }

                // postproc
                if (scale == 4)
                {
                    std::vector<ncnn::VkMat> bindings(10);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu[0];
                    bindings[2] = out_tile_gpu[1];
                    bindings[3] = out_tile_gpu[2];
                    bindings[4] = out_tile_gpu[3];
                    bindings[5] = out_tile_gpu[4];
                    bindings[6] = out_tile_gpu[5];
                    bindings[7] = out_tile_gpu[6];
                    bindings[8] = out_tile_gpu[7];
                    bindings[9] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(14);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu[0].w;
                    constants[4].i = out_tile_gpu[0].h;
                    constants[5].i = out_tile_gpu[0].cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = out_tile_gpu[0];
                    bindings[1] = out_tile_gpu[1];
                    bindings[2] = out_tile_gpu[2];
                    bindings[3] = out_tile_gpu[3];
                    bindings[4] = out_tile_gpu[4];
                    bindings[5] = out_tile_gpu[5];
                    bindings[6] = out_tile_gpu[6];
                    bindings[7] = out_tile_gpu[7];
                    bindings[8] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(9);
                    constants[0].i = out_tile_gpu[0].w;
                    constants[1].i = out_tile_gpu[0].h;
                    constants[2].i = out_tile_gpu[0].cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = CHANNELS;
                    
                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu;
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    ex.extract("out0", out_tile_gpu, cmd);
                }

                // postproc
                if (scale == 4)
                {
                    std::vector<ncnn::VkMat> bindings(3);
                    bindings[0] = in_gpu;
                    bindings[1] = out_tile_gpu;
                    bindings[2] = out_gpu;

                    std::vector<ncnn::vk_constant_type> constants(14);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = out_tile_gpu.w;
                    constants[4].i = out_tile_gpu.h;
                    constants[5].i = out_tile_gpu.cstep;
                    constants[6].i = out_gpu.w;
                    constants[7].i = out_gpu.h;
                    constants[8].i = out_gpu.cstep;
                    constants[9].i = xi * TILE_SIZE_X;
                    constants[10].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[11].i = xi * TILE_SIZE_X * scale;
                    constants[12].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[13].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_4x_postproc, bindings, constants, dispatcher);
                }
                else
                {
                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = out_tile_gpu;
                    bindings[1] = out_gpu;

                    std::vector<ncnn::vk_constant_type>constants(9);
                    constants[0].i = out_tile_gpu.w;
                    constants[1].i = out_tile_gpu.h;
                    constants[2].i = out_tile_gpu.cstep;
                    constants[3].i = out_gpu.w;
                    constants[4].i = out_gpu.h;
                    constants[5].i = out_gpu.cstep;
                    constants[6].i = xi * TILE_SIZE_X * scale;
                    constants[7].i = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    constants[8].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = std::min(TILE_SIZE_X * scale, out_gpu.w - xi * TILE_SIZE_X * scale);
                    dispatcher.h = out_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_postproc, bindings, constants, dispatcher);
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        // download
        {
            ncnn::Mat out;

            cmd.record_clone(out_gpu, out, opt);
            cmd.submit_and_wait();

            if (!(opt.use_fp16_storage && opt.use_int8_storage))
            {
                const float* out_tile_r = out.channel(0);
                const float* out_tile_g = out.channel(1);
                const float* out_tile_b = out.channel(2);

                float* dr = dstpR + yi * TILE_SIZE_Y * scale * dst_stride;
                float* dg = dstpG + yi * TILE_SIZE_Y * scale * dst_stride;
                float* db = dstpB + yi * TILE_SIZE_Y * scale * dst_stride;

                for (int y = 0; y < out.h; y++)
                {
                    for (int x = 0; x < out.w; x++)
                    {
                        dr[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_r[out.w * y + x] / 255.f));
                        dg[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_g[out.w * y + x] / 255.f));
                        db[dst_stride * y + x] = std::min(1.f, std::max(0.f, out_tile_b[out.w * y + x] / 255.f));
                    }
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_se_sync_gap(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache) const
{
    const int TILE_SIZE_X = tilesize_x;
    const int TILE_SIZE_Y = tilesize_y;

    // each tile 400x400
    const int xtiles = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector<std::vector<ncnn::VkMat>> feats(names.size());
    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            {
                for (size_t i = 0; i < names.size(); i++)
                {
                    if (tta_mode)
                    {
                        for (int ti = 0; ti < 8; ti++)
                        {
                            ncnn::VkMat feat;
                            cache.load(yi, xi, ti, names[i], feat);

                            feats[i].push_back(feat);
                        }
                    }
                    else
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        feats[i].push_back(feat);
                    }
                }
            }
        }
    }

    const int tiles = ytiles * xtiles * (tta_mode ? 8 : 1);

    ncnn::VkCompute cmd(vkdev);

    // download
    std::vector<std::vector<ncnn::Mat>> feats_cpu(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        feats_cpu[i].resize(tiles);

        for (int j = 0; j < tiles; j++)
        {
            cmd.record_download(feats[i][j], feats_cpu[i][j], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    // global average
    // upload
    std::vector<ncnn::VkMat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        for (int j = 0; j < tiles; j++)
        {
            if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && feats_cpu[i][j].elembits() == 16)
            {
                ncnn::Mat feat_fp32;
                ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                feats_cpu[i][j] = feat_fp32;
            }

            if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1)
            {
                ncnn::Mat feat_cpu_unpacked;
                ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                feats_cpu[i][j] = feat_cpu_unpacked;
            }
        }

        // handle feats_cpu[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats_cpu[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++)
            {
                const ncnn::Mat f = feats_cpu[i][j];

                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] += f[k];
                }
            }

            for (int k = 0; k < len; k++)
            {
                avgfeat[k] /= tiles;
            }

            cmd.record_upload(avgfeat, avgfeats[i], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    for (int yi = 0; yi < ytiles; yi++)
    {
        for (int xi = 0; xi < xtiles; xi++)
        {
            {
                for (size_t i = 0; i < names.size(); i++)
                {
                    if (tta_mode)
                    {
                        for (int ti = 0; ti < 8; ti++)
                        {
                            cache.save(yi, xi, ti, names[i], avgfeats[i]);
                        }
                    }
                    else
                    {
                        cache.save(yi, xi, 0, names[i], avgfeats[i]);
                    }
                }
            }
        }
    }

    return 0;
}

int RealCUGAN::process_se_very_rough_stage0(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const std::vector<std::string>& outnames, const ncnn::Option& opt, FeatureCache& cache) const
{
    const int TILE_SIZE_X = 32;  // set as const value for stage0()
    const int TILE_SIZE_Y = 32;

    // each tile 400x400 
    const int xtiles = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    const size_t in_out_tile_elemsize = opt.use_fp16_storage ? 2u : 4u;

    //#pragma omp parallel for num_threads(2)
    for (int yi = 0; yi + 2 < ytiles; yi += 3)
    {
        const int tile_h_nopad = std::min((yi + 1) * TILE_SIZE_Y, height) - yi * TILE_SIZE_Y;

        int prepadding_bottom = prepadding;
        if (scale == 3)
        {
            prepadding_bottom += (tile_h_nopad + 3) / 4 * 4 - tile_h_nopad;
        }
        if (scale == 2 || scale == 4)
        {
            prepadding_bottom += (tile_h_nopad + 1) / 2 * 2 - tile_h_nopad;
        }

        int in_tile_y0 = std::max(yi * TILE_SIZE_Y - prepadding, 0);
        int in_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y + prepadding_bottom, height);
        const int in_tile_w = width;
        const int in_tile_h = in_tile_y1 - in_tile_y0;

        ncnn::Mat in(in_tile_w, in_tile_h, CHANNELS, sizeof(float));

        float* in_tile_r = in.channel(0);
        float* in_tile_g = in.channel(1);
        float* in_tile_b = in.channel(2);

        const float* sr = srcpR + in_tile_y0 * src_stride;
        const float* sg = srcpG + in_tile_y0 * src_stride;
        const float* sb = srcpB + in_tile_y0 * src_stride;

        for (int y = 0; y < in_tile_h; y++)
        {
            for (int x = 0; x < in_tile_w; x++)
            {
                in_tile_r[in_tile_w * y + x] = sr[src_stride * y + x] * 255.f;
                in_tile_g[in_tile_w * y + x] = sg[src_stride * y + x] * 255.f;
                in_tile_b[in_tile_w * y + x] = sb[src_stride * y + x] * 255.f;
            }
        }

        ncnn::VkCompute cmd(vkdev);

        // upload
        ncnn::VkMat in_gpu;
        {
            cmd.record_clone(in, in_gpu, opt);

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        int out_tile_y0 = std::max(yi * TILE_SIZE_Y, 0);
        int out_tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height);

        ncnn::VkMat out_gpu;
        out_gpu.create(width * scale, (out_tile_y1 - out_tile_y0) * scale, CHANNELS, sizeof(float), opt.blob_vkallocator);

        for (int xi = 0; xi + 2 < xtiles; xi += 3)
        {
            const int tile_w_nopad = std::min((xi + 1) * TILE_SIZE_X, width) - xi * TILE_SIZE_X;

            int prepadding_right = prepadding;
            if (scale == 3)
            {
                prepadding_right += (tile_w_nopad + 3) / 4 * 4 - tile_w_nopad;
            }
            if (scale == 2 || scale == 4)
            {
                prepadding_right += (tile_w_nopad + 1) / 2 * 2 - tile_w_nopad;
            }

            if (tta_mode)
            {
                // preproc
                ncnn::VkMat in_tile_gpu[8];

                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu[0].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[1].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[2].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[3].create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[4].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[5].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[6].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    in_tile_gpu[7].create(tile_y1 - tile_y0, tile_x1 - tile_x0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);
                    
                    std::vector<ncnn::VkMat> bindings(9);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu[0];
                    bindings[2] = in_tile_gpu[1];
                    bindings[3] = in_tile_gpu[2];
                    bindings[4] = in_tile_gpu[3];
                    bindings[5] = in_tile_gpu[4];
                    bindings[6] = in_tile_gpu[5];
                    bindings[7] = in_tile_gpu[6];
                    bindings[8] = in_tile_gpu[7];

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu[0].w;
                    constants[4].i = in_tile_gpu[0].h;
                    constants[5].i = in_tile_gpu[0].cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu[0].w;
                    dispatcher.h = in_tile_gpu[0].h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                ncnn::VkMat out_tile_gpu[8];
                for (int ti = 0; ti < 8; ti++)
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu[ti]);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, ti, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, ti, outnames[i], feat);
                    }
                }
            }
            else
            {
                // preproc
                ncnn::VkMat in_tile_gpu;
                {
                    // crop tile
                    int tile_x0 = xi * TILE_SIZE_X - prepadding;
                    int tile_x1 = std::min((xi + 1) * TILE_SIZE_X, width) + prepadding_right;
                    int tile_y0 = yi * TILE_SIZE_Y - prepadding;
                    int tile_y1 = std::min((yi + 1) * TILE_SIZE_Y, height) + prepadding_bottom;

                    in_tile_gpu.create(tile_x1 - tile_x0, tile_y1 - tile_y0, CHANNELS, in_out_tile_elemsize, 1, opt.blob_vkallocator);

                    std::vector<ncnn::VkMat> bindings(2);
                    bindings[0] = in_gpu;
                    bindings[1] = in_tile_gpu;

                    std::vector<ncnn::vk_constant_type> constants(11);
                    constants[0].i = in_gpu.w;
                    constants[1].i = in_gpu.h;
                    constants[2].i = in_gpu.cstep;
                    constants[3].i = in_tile_gpu.w;
                    constants[4].i = in_tile_gpu.h;
                    constants[5].i = in_tile_gpu.cstep;
                    constants[6].i = prepadding;
                    constants[7].i = prepadding;
                    constants[8].i = xi * TILE_SIZE_X;
                    constants[9].i = std::min(yi * TILE_SIZE_Y, prepadding);
                    constants[10].i = CHANNELS;

                    ncnn::VkMat dispatcher;
                    dispatcher.w = in_tile_gpu.w;
                    dispatcher.h = in_tile_gpu.h;
                    dispatcher.c = CHANNELS;

                    cmd.record_pipeline(realcugan_preproc, bindings, constants, dispatcher);
                }

                // realcugan
                {
                    ncnn::Extractor ex = net.create_extractor();

                    ex.set_blob_vkallocator(opt.blob_vkallocator);
                    ex.set_workspace_vkallocator(opt.blob_vkallocator);
                    ex.set_staging_vkallocator(opt.staging_vkallocator);

                    ex.input("in0", in_tile_gpu);

                    for (size_t i = 0; i < names.size(); i++)
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        ex.input(names[i].c_str(), feat);
                    }

                    for (size_t i = 0; i < outnames.size(); i++)
                    {
                        ncnn::VkMat feat;
                        ex.extract(outnames[i].c_str(), feat, cmd);

                        cache.save(yi, xi, 0, outnames[i], feat);
                    }
                }
            }

            if (xtiles > 1)
            {
                cmd.submit_and_wait();
                cmd.reset();
            }
        }

        cmd.submit_and_wait();
        cmd.reset();
    }

    return 0;
}

int RealCUGAN::process_se_very_rough_sync_gap(const float* srcpR, const float* srcpG, const float* srcpB, int width, int height, int src_stride, const std::vector<std::string>& names, const ncnn::Option& opt, FeatureCache& cache) const
{
    const int TILE_SIZE_X = 32;
    const int TILE_SIZE_Y = 32;

    // each tile 400x400
    const int xtiles = (width + TILE_SIZE_X - 1) / TILE_SIZE_X;
    const int ytiles = (height + TILE_SIZE_Y - 1) / TILE_SIZE_Y;

    std::vector<std::vector<ncnn::VkMat>> feats(names.size());
    for (int yi = 0; yi + 2 < ytiles; yi += 3)
    {
        for (int xi = 0; xi + 2 < xtiles; xi += 3)
        {
            {
                for (size_t i = 0; i < names.size(); i++)
                {
                    if (tta_mode)
                    {
                        for (int ti = 0; ti < 8; ti++)
                        {
                            ncnn::VkMat feat;
                            cache.load(yi, xi, ti, names[i], feat);

                            feats[i].push_back(feat);
                        }
                    }
                    else
                    {
                        ncnn::VkMat feat;
                        cache.load(yi, xi, 0, names[i], feat);

                        feats[i].push_back(feat);
                    }
                }
            }
        }
    }

    const int tiles = (ytiles / 3) * (xtiles / 3) * (tta_mode ? 8 : 1);

    ncnn::VkCompute cmd(vkdev);

    // download
    std::vector<std::vector<ncnn::Mat>> feats_cpu(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        feats_cpu[i].resize(tiles);

        for (int j = 0; j < tiles; j++)
        {
            cmd.record_download(feats[i][j], feats_cpu[i][j], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    // global average
    // upload
    std::vector<ncnn::VkMat> avgfeats(names.size());
    for (size_t i = 0; i < names.size(); i++)
    {
        for (int j = 0; j < tiles; j++)
        {
            if (opt.use_fp16_storage && ncnn::cpu_support_arm_asimdhp() && feats_cpu[i][j].elembits() == 16)
            {
                ncnn::Mat feat_fp32;
                ncnn::cast_float16_to_float32(feats_cpu[i][j], feat_fp32, opt);
                feats_cpu[i][j] = feat_fp32;
            }

            if (opt.use_packing_layout && feats_cpu[i][j].elempack != 1)
            {
                ncnn::Mat feat_cpu_unpacked;
                ncnn::convert_packing(feats_cpu[i][j], feat_cpu_unpacked, 1, opt);
                feats_cpu[i][j] = feat_cpu_unpacked;
            }
        }

        // handle feats_cpu[i] vector
        {
            ncnn::Mat avgfeat;
            avgfeat.create_like(feats_cpu[i][0]);
            avgfeat.fill(0.f);

            int len = avgfeat.total();

            for (int j = 0; j < tiles; j++)
            {
                const ncnn::Mat f = feats_cpu[i][j];

                for (int k = 0; k < len; k++)
                {
                    avgfeat[k] += f[k];
                }
            }

            for (int k = 0; k < len; k++)
            {
                avgfeat[k] /= tiles;
            }

            cmd.record_upload(avgfeat, avgfeats[i], opt);
        }
    }

    cmd.submit_and_wait();
    cmd.reset();

    for (int yi = 0; yi + 2 < ytiles; yi += 3)
    {
        for (int xi = 0; xi + 2 < xtiles; xi += 3)
        {
            {
                for (size_t i = 0; i < names.size(); i++)
                {
                    if (tta_mode)
                    {
                        for (int ti = 0; ti < 8; ti++)
                        {
                            cache.save(yi, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi, xi + 2, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi + 1, xi + 2, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi + 1, ti, names[i], avgfeats[i]);
                            cache.save(yi + 2, xi + 2, ti, names[i], avgfeats[i]);
                        }
                    }
                    else
                    {
                        cache.save(yi, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi, xi + 2, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi + 1, xi + 2, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi + 1, 0, names[i], avgfeats[i]);
                        cache.save(yi + 2, xi + 2, 0, names[i], avgfeats[i]);
                    }
                }
            }
        }
    }

    return 0;
}
