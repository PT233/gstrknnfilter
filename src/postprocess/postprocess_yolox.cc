// YOLOX postprocess - 3 outputs, grid-based, exp() for box w/h (no anchors)
// Each output: [1, 85, H, W] - 4 box + 1 obj + 80 class per cell

#include "postprocess_common.h"
#include "postprocess_impl.h"
#include "../rknnprocess.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <set>
#include <vector>

static char* labels[OBJ_CLASS_NUM];
static int labels_loaded = 0;

static int loadLabelName(const char* locationFilename, char* label[]);
static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
    float xmin1, float ymin1, float xmax1, float ymax1);
static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
    std::vector<int>& order, int filterId, float threshold);
static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices);

static char* readLine(FILE* fp, char* buffer, int* len);
static int readLines(const char* fileName, char* lines[], int max_line);

static char* readLine(FILE* fp, char* buffer, int* len) {
    int ch, i = 0;
    size_t buff_len = 0;
    buffer = (char*)malloc(1);
    if (!buffer) return NULL;
    while ((ch = fgetc(fp)) != '\n' && ch != EOF) {
        buff_len++;
        void* tmp = realloc(buffer, buff_len + 1);
        if (!tmp) { free(buffer); return NULL; }
        buffer = (char*)tmp;
        buffer[i++] = (char)ch;
    }
    buffer[i] = '\0';
    *len = (int)buff_len;
    if (ch == EOF && (i == 0 || ferror(fp))) { free(buffer); return NULL; }
    return buffer;
}

static int readLines(const char* fileName, char* lines[], int max_line) {
    FILE* file = fopen(fileName, "r");
    char* s = NULL;
    int i = 0, n = 0;
    if (!file) return -1;
    while ((s = readLine(file, s, &n)) != NULL) {
        lines[i++] = s;
        if (i >= max_line) break;
    }
    fclose(file);
    return i;
}

static int loadLabelName(const char* locationFilename, char* label[]) {
    if (labels_loaded) return 0;
    int ret = readLines(locationFilename, label, OBJ_CLASS_NUM);
    if (ret > 0) labels_loaded = 1;
    return ret < 0 ? -1 : 0;
}

static float CalculateOverlap(float xmin0, float ymin0, float xmax0, float ymax0,
    float xmin1, float ymin1, float xmax1, float ymax1) {
    float w = fmax(0.f, fmin(xmax0, xmax1) - fmax(xmin0, xmin1) + 1.0f);
    float h = fmax(0.f, fmin(ymax0, ymax1) - fmax(ymin0, ymin1) + 1.0f);
    float i = w * h;
    float u = (xmax0 - xmin0 + 1.0f) * (ymax0 - ymin0 + 1.0f) +
        (xmax1 - xmin1 + 1.0f) * (ymax1 - ymin1 + 1.0f) - i;
    return u <= 0.f ? 0.f : (i / u);
}

static int nms(int validCount, std::vector<float>& outputLocations, std::vector<int> classIds,
    std::vector<int>& order, int filterId, float threshold) {
    for (int i = 0; i < validCount; ++i) {
        int n = order[i];
        if (n == -1 || classIds[n] != filterId) continue;
        for (int j = i + 1; j < validCount; ++j) {
            int m = order[j];
            if (m == -1 || classIds[m] != filterId) continue;
            float xmin0 = outputLocations[n * 4 + 0];
            float ymin0 = outputLocations[n * 4 + 1];
            float xmax0 = outputLocations[n * 4 + 0] + outputLocations[n * 4 + 2];
            float ymax0 = outputLocations[n * 4 + 1] + outputLocations[n * 4 + 3];
            float xmin1 = outputLocations[m * 4 + 0];
            float ymin1 = outputLocations[m * 4 + 1];
            float xmax1 = outputLocations[m * 4 + 0] + outputLocations[m * 4 + 2];
            float ymax1 = outputLocations[m * 4 + 1] + outputLocations[m * 4 + 3];
            float iou = CalculateOverlap(xmin0, ymin0, xmax0, ymax0, xmin1, ymin1, xmax1, ymax1);
            if (iou > threshold) order[j] = -1;
        }
    }
    return 0;
}

static int quick_sort_indice_inverse(std::vector<float>& input, int left, int right, std::vector<int>& indices) {
    if (left >= right) return left;
    float key = input[left];
    int key_index = indices[left];
    int low = left, high = right;
    while (low < high) {
        while (low < high && input[high] <= key) high--;
        input[low] = input[high];
        indices[low] = indices[high];
        while (low < high && input[low] >= key) low++;
        input[high] = input[low];
        indices[high] = indices[low];
    }
    input[low] = key;
    indices[low] = key_index;
    quick_sort_indice_inverse(input, left, low - 1, indices);
    quick_sort_indice_inverse(input, low + 1, right, indices);
    return low;
}

static int process_i8_yolox(int8_t* input, int grid_h, int grid_w, int stride,
    std::vector<float>& boxes, std::vector<float>& objProbs, std::vector<int>& classId,
    float threshold, int32_t zp, float scale) {
    int validCount = 0;
    int grid_len = grid_h * grid_w;
    int8_t thres_i8 = (int8_t)((threshold / scale) + zp);
    if (thres_i8 > 127) thres_i8 = 127;
    if (thres_i8 < -128) thres_i8 = -128;

    for (int i = 0; i < grid_h; i++) {
        for (int j = 0; j < grid_w; j++) {
            int8_t box_confidence = input[4 * grid_len + i * grid_w + j];
            if (box_confidence < thres_i8) continue;

            int offset = i * grid_w + j;
            int8_t* in_ptr = input + offset;

            int8_t maxClassProbs = in_ptr[5 * grid_len];
            int maxClassId = 0;
            for (int k = 1; k < OBJ_CLASS_NUM; k++) {
                int8_t prob = in_ptr[(5 + k) * grid_len];
                if (prob > maxClassProbs) {
                    maxClassId = k;
                    maxClassProbs = prob;
                }
            }
            if (maxClassProbs < thres_i8) continue;

            float box_x = deqnt_affine_to_f32(*in_ptr, zp, scale);
            float box_y = deqnt_affine_to_f32(in_ptr[grid_len], zp, scale);
            float box_w = deqnt_affine_to_f32(in_ptr[2 * grid_len], zp, scale);
            float box_h = deqnt_affine_to_f32(in_ptr[3 * grid_len], zp, scale);
            box_x = (box_x + j) * (float)stride;
            box_y = (box_y + i) * (float)stride;
            box_w = expf(box_w) * stride;
            box_h = expf(box_h) * stride;
            box_x -= (box_w / 2.0f);
            box_y -= (box_h / 2.0f);

            objProbs.push_back(deqnt_affine_to_f32(maxClassProbs, zp, scale) *
                deqnt_affine_to_f32(box_confidence, zp, scale));
            classId.push_back(maxClassId);
            boxes.push_back(box_x);
            boxes.push_back(box_y);
            boxes.push_back(box_w);
            boxes.push_back(box_h);
            validCount++;
        }
    }
    return validCount;
}

int postprocess_yolox(struct _RknnProcess* rknn_process,
    float box_conf_threshold, float nms_threshold,
    std::vector<int32_t>& qnt_zps, std::vector<float>& qnt_scales,
    detect_result_group_t* group, char* label_path) {
    if (rknn_process->io_num.n_output < 3) return -1;
    if (loadLabelName(label_path, labels) < 0) return -1;

    memset(group, 0, sizeof(detect_result_group_t));
    std::vector<float> filterBoxes;
    std::vector<float> objProbs;
    std::vector<int> classId;
    int validCount = 0;
    int model_in_h = rknn_process->model_height;
    int model_in_w = rknn_process->model_width;

    for (int i = 0; i < 3; i++) {
        int grid_h = rknn_process->output_attrs[i].dims[2];
        int grid_w = rknn_process->output_attrs[i].dims[3];
        int stride = model_in_h / grid_h;
        int32_t zp = qnt_zps[i];
        float scale = qnt_scales[i];

        validCount += process_i8_yolox(
            (int8_t*)rknn_process->outputs[i].buf,
            grid_h, grid_w, stride,
            filterBoxes, objProbs, classId,
            box_conf_threshold, zp, scale);
    }

    if (validCount <= 0) return 0;

    std::vector<int> indexArray;
    for (int i = 0; i < validCount; i++) indexArray.push_back(i);
    quick_sort_indice_inverse(objProbs, 0, validCount - 1, indexArray);

    std::set<int> class_set(classId.begin(), classId.end());
    for (int c : class_set) {
        nms(validCount, filterBoxes, classId, indexArray, c, nms_threshold);
    }

    int last_count = 0;
    for (int i = 0; i < validCount; i++) {
        if (indexArray[i] == -1 || last_count >= OBJ_NUMB_MAX_SIZE) continue;
        int n = indexArray[i];
        float x1 = filterBoxes[n * 4 + 0] - rknn_process->pads.left;
        float y1 = filterBoxes[n * 4 + 1] - rknn_process->pads.top;
        float x2 = x1 + filterBoxes[n * 4 + 2];
        float y2 = y1 + filterBoxes[n * 4 + 3];
        int id = classId[n];

        group->results[last_count].box.left = (int)(clamp(x1, 0.f, (float)model_in_w) / rknn_process->scale_w);
        group->results[last_count].box.top = (int)(clamp(y1, 0.f, (float)model_in_h) / rknn_process->scale_h);
        group->results[last_count].box.right = (int)(clamp(x2, 0.f, (float)model_in_w) / rknn_process->scale_w);
        group->results[last_count].box.bottom = (int)(clamp(y2, 0.f, (float)model_in_h) / rknn_process->scale_h);
        group->results[last_count].prop = objProbs[i];
        if (labels[id]) strncpy(group->results[last_count].name, labels[id], OBJ_NAME_MAX_SIZE - 1);
        group->results[last_count].name[OBJ_NAME_MAX_SIZE - 1] = '\0';
        last_count++;
    }
    group->count = last_count;
    return 0;
}

void deinit_postprocess_yolox(void) {
    for (int i = 0; i < OBJ_CLASS_NUM; i++) {
        if (labels[i]) { free(labels[i]); labels[i] = NULL; }
    }
    labels_loaded = 0;
}
