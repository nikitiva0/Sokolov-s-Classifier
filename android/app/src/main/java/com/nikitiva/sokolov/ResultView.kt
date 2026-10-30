package com.nikitiva.sokolov

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.util.AttributeSet
import android.view.View
import kotlin.math.min

/** Bounding boxes are drawn over the exact analyzed frame, never a later preview frame. */
class ResultView(context: Context, attrs: AttributeSet? = null): View(context,attrs) {
    private var image: Bitmap?=null
    private var result: Engine.Result?=null
    private val paint=Paint(Paint.ANTI_ALIAS_FLAG)
    fun show(bitmap: Bitmap, analysis: Engine.Result) {
        val previous=image
        image=bitmap; result=analysis
        if(previous!==bitmap)previous?.recycle()
        invalidate()
    }
    override fun onDraw(canvas: Canvas) {
        super.onDraw(canvas)
        canvas.drawColor(Color.rgb(18,24,35))
        val bitmap=image ?: return
        val scale=min(width.toFloat()/bitmap.width,height.toFloat()/bitmap.height)
        val dx=(width-bitmap.width*scale)/2; val dy=(height-bitmap.height*scale)/2
        canvas.save(); canvas.translate(dx,dy); canvas.scale(scale,scale)
        canvas.drawBitmap(bitmap,0f,0f,null)
        for(face in result?.faces.orEmpty()) {
            val color=if(face.error!=null)Color.YELLOW else if(face.positive)Color.rgb(50,230,120) else Color.rgb(255,170,70)
            paint.color=color; paint.style=Paint.Style.STROKE; paint.strokeWidth=3/scale
            canvas.drawRect(face.box[0],face.box[1],face.box[2],face.box[3],paint)
            paint.style=Paint.Style.FILL; paint.textSize=16*resources.displayMetrics.scaledDensity/scale
            val label=face.error ?: "${if(face.positive) "Соколов" else "Не Соколов"} · %.3f".format(face.score)
            val x=face.box[0].coerceIn(0f,bitmap.width.toFloat())
            val y=(face.box[1]-6/scale).coerceAtLeast(paint.textSize)
            canvas.drawText(label,x,y,paint)
        }
        canvas.restore()
    }
}
