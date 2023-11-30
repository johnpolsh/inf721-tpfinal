package com.jopzoli.objectdetection

import android.graphics.Rect
import java.util.Arrays
import kotlin.math.max
import kotlin.math.min

class Detection(
    val rect: Rect,
    val classIdx: Int,
    val score: Float
) {
}

class PrePostProcessor {

    companion object {
        const val INPUT_WIDTH = 640
        const val INPUT_HEIGHT = 640

        val NO_MEAN_RGB = floatArrayOf(0f, 0f, 0f)
        val NO_STD_RGB = floatArrayOf(1f, 1f, 1f)

        private const val OUTPUT_ROW = 25200
        private const val OUTPUT_COL = 85
        private const val THRESHOLD = .6f
        private const val TIMEOUT = 15

        private fun nonMaxSuppression(
            boxes: ArrayList<Detection>
        ): ArrayList<Detection> {
            boxes.sortWith { o1, o2 -> o1.score.compareTo(o2.score) }
            val selected = ArrayList<Detection>()
            val active = BooleanArray(boxes.size)
            Arrays.fill(active, true)
            var numActive = active.size

            var done = false
            var i = 0
            while (i < boxes.size && !done) {
                if (active[i]) {
                    val boxA = boxes[i]
                    selected.add(boxA)
                    if (selected.size >= TIMEOUT) break
                    for (j in i + 1 until boxes.size) {
                        if (active[j]) {
                            val boxB = boxes[j]
                            if (iou(boxA.rect, boxB.rect) > THRESHOLD) {
                                active[j] = false
                                numActive -= 1
                                if (numActive <= 0) {
                                    done = true
                                    break
                                }
                            }
                        }
                    }
                }
                i++
            }

            return selected
        }

        private fun iou(a: Rect, b: Rect): Float {
            val areaA = ((a.right - a.left) * (a.bottom - a.top)).toFloat()
            if (areaA <= 0f)
                return 0f
            val areaB = ((b.right - b.left) * (b.bottom - b.top)).toFloat()
            if (areaB <= 0f)
                return 0f
            val intersectionMinX = max(a.left, b.left).toFloat()
            val intersectionMinY = max(a.top, b.top).toFloat()
            val intersectionMaxX = min(a.right, b.right).toFloat()
            val intersectionMaxY = min(a.bottom, b.bottom).toFloat()
            val intersectionArea = max(
                intersectionMaxY - intersectionMinY, 0f
            ) *
                    max(intersectionMaxX - intersectionMinX, 0f)

            return intersectionArea / (areaA + areaB - intersectionArea)
        }

        fun outputsToNMSPredictions(
            outputs: FloatArray,
            imgScaleX: Float,
            imgScaleY: Float,
            ivScaleX: Float,
            ivScaleY: Float,
            startX: Float,
            startY: Float
        ): ArrayList<Detection> {
            val detections = ArrayList<Detection>()
            for (i in 0 until OUTPUT_ROW) {
                if (outputs[i * OUTPUT_COL + 4] > THRESHOLD) {
                    val x = outputs[i * OUTPUT_COL]
                    val y = outputs[i * OUTPUT_COL + 1]
                    val w = outputs[i * OUTPUT_COL + 2]
                    val h = outputs[i * OUTPUT_COL + 3]
                    val left = imgScaleX * (x - w / 2)
                    val top = imgScaleY * (y - h / 2)
                    val right = imgScaleX * (x + w / 2)
                    val bottom = imgScaleY * (y + h / 2)
                    var max = outputs[i * OUTPUT_COL + 5]
                    var cls = 0
                    for (j in 0 until OUTPUT_COL - 5) {
                        if (outputs[i * OUTPUT_COL + 5 + j] > max) {
                            max = outputs[i * OUTPUT_COL + 5 + j]
                            cls = j
                        }
                    }
                    val rect = Rect(
                        (startX + ivScaleX * left).toInt(),
                        (startY + top * ivScaleY).toInt(),
                        (startX + ivScaleX * right).toInt(),
                        (startY + ivScaleY * bottom).toInt()
                    )
                    val detection = Detection(rect, cls, outputs[i * OUTPUT_COL + 4])
                    detections.add(detection)
                }
            }

            return nonMaxSuppression(detections)
        }

    }
}