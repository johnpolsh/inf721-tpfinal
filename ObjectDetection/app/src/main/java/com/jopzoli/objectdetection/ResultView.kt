package com.jopzoli.objectdetection

import android.content.Context
import android.graphics.Canvas
import android.graphics.Paint
import android.graphics.Path
import android.graphics.RectF
import android.util.AttributeSet
import android.view.View
import androidx.core.content.ContextCompat
import java.util.LinkedList

class DetectionView(context: Context?, attrs: AttributeSet?) : View(context, attrs) {
    private val boxPaint = Paint()
    private val labelBkPaint = Paint()
    private val labelPaint = Paint()
    private val labelBkPath = Path()
    private var detections: List<Detection> = LinkedList<Detection>()

    init {
        boxPaint.color = ContextCompat.getColor(context!!, R.color.detection_box_stroke)
        boxPaint.strokeWidth = 5f
        boxPaint.style = Paint.Style.STROKE

        labelBkPaint.color = ContextCompat.getColor(context, R.color.detection_label_bk)
        labelBkPaint.textSize = 50f
        labelBkPaint.style = Paint.Style.FILL

        labelPaint.color = ContextCompat.getColor(context, R.color.detection_label_text)
        labelPaint.textSize = 50f
        labelPaint.style = Paint.Style.FILL
    }

    override fun draw(canvas: Canvas) {
        super.draw(canvas)

        for (detection in detections) {
            canvas.drawRect(detection.rect, boxPaint)

            val rect = RectF(
                detection.rect.left.toFloat(),
                detection.rect.top.toFloat(),
                (detection.rect.left + TEXT_WIDTH).toFloat(),
                (detection.rect.top + TEXT_HEIGHT).toFloat()
                )
            labelBkPath.addRect(rect, Path.Direction.CW)
            canvas.drawPath(labelBkPath, labelBkPaint)

//            val label = PrePostProcessor.labels[detection.classIdx]
            val label = "<none>"
            val score = detection.score
            canvas.drawText(
                String.format("%s %.2f", label, score),
                (detection.rect.left + TEXT_LEFT).toFloat(),
                (detection.rect.top + TEXT_TOP).toFloat(),
                labelPaint)
        }
    }

    fun setDetectionResults(results: ArrayList<Detection>) {
        detections = results
    }

    companion object {
        const val TEXT_LEFT = 40
        const val TEXT_TOP = 35
        const val TEXT_WIDTH = 260
        const val TEXT_HEIGHT = 50
    }
}