#pragma once
#include <opencv2/opencv.hpp>
/**
 * @brief  缩放图像
 * @note
 * @param  src: 原始图像设备内存地址
 * @param  src_wdith: 原始图像宽
 * @param  src_height: 原始图像高
 * @param  width_step: 原始图像一行数据占用字节数
 * @param  dst_buf: 目标图像设备内存地址
 * @param  dst_width: 图像宽
 * @param  dst_height: 图像宽
 * @retval
 */
bool resize_image(const void *src, int src_wdith, int src_height,
                  int width_step, void *dst_buf, int dst_width, int dst_height);

bool padding_image(const void *src, int src_wdith, int src_height,
                   int width_step, unsigned char pad_vaule[3], void *dst_buf,
                   int dst_wid, int dst_hei);

bool normlize_image(float *src, int src_wdith, int src_height, int channel,
                    float mean[3], float std_val[3]);

bool crop_image(const unsigned char *src, int src_wdith, int src_height,
                int src_channel, int width_step, const cv::Rect &roi,
                unsigned char *dst_buffer);