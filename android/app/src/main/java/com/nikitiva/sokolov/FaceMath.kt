package com.nikitiva.sokolov

import kotlin.math.max
import kotlin.math.min

/** Similarity least squares matching the ArcFace five-point template; no Android dependency. */
object FaceMath {
    val template = doubleArrayOf(38.2946,51.6963,73.5318,51.5014,56.0252,71.7366,
        41.5493,92.3655,70.7299,92.2041).map { it * 2 }.toDoubleArray()
    // Returns x'=a*x-b*y+tx, y'=b*x+a*y+ty. This is the same non-reflective
    // least-squares similarity family used by skimage SimilarityTransform.
    fun similarity(source: FloatArray, target: DoubleArray = template): DoubleArray {
        require(source.size == 10 && source.all { it.isFinite() })
        val sx = (0..4).sumOf { source[2*it].toDouble() } / 5
        val sy = (0..4).sumOf { source[2*it+1].toDouble() } / 5
        val tx = (0..4).sumOf { target[2*it] } / 5
        val ty = (0..4).sumOf { target[2*it+1] } / 5
        var denominator=0.0; var dot=0.0; var cross=0.0
        for (i in 0..4) {
            val x=source[2*i]-sx; val y=source[2*i+1]-sy
            val u=target[2*i]-tx; val v=target[2*i+1]-ty
            denominator+=x*x+y*y; dot+=x*u+y*v; cross+=x*v-y*u
        }
        require(denominator > 1e-8) { "Вырожденные ключевые точки" }
        val a=dot/denominator; val b=cross/denominator
        return doubleArrayOf(a,b,tx-a*sx+b*sy,ty-b*sx-a*sy)
    }
    data class Detection(val box: FloatArray, val landmarks: FloatArray, val confidence: Float)
    fun iou(a: FloatArray, b: FloatArray): Float {
        val intersection=max(0f,min(a[2],b[2])-max(a[0],b[0])+1)*
            max(0f,min(a[3],b[3])-max(a[1],b[1])+1)
        val union=(a[2]-a[0]+1)*(a[3]-a[1]+1)+(b[2]-b[0]+1)*(b[3]-b[1]+1)-intersection
        return if (union>0) intersection/union else 0f
    }
    fun nms(candidates: List<Detection>, threshold: Float): List<Detection> {
        val keep=mutableListOf<Detection>()
        for (item in candidates.sortedByDescending { it.confidence }) {
            if (keep.none { iou(it.box,item.box)>threshold }) keep+=item
        }
        return keep
    }
    fun decode(outputs: List<FloatArray>, inputSize: Int, scale: Float,
               threshold: Float, nms: Float): List<Detection> {
        require(outputs.size==9 && scale>0)
        val candidates=mutableListOf<Detection>()
        for ((level,stride) in listOf(8,16,32).withIndex()) {
            val width=inputSize/stride; val count=width*width*2
            val scores=outputs[level]; val boxes=outputs[level+3]; val points=outputs[level+6]
            require(scores.size==count && boxes.size==count*4 && points.size==count*10)
            for (anchor in 0 until count) {
                if (scores[anchor]<threshold) continue
                val index=anchor/2
                val cx=(index%width)*stride.toFloat(); val cy=(index/width)*stride.toFloat()
                val box=floatArrayOf((cx-boxes[anchor*4]*stride)/scale,
                    (cy-boxes[anchor*4+1]*stride)/scale,
                    (cx+boxes[anchor*4+2]*stride)/scale,(cy+boxes[anchor*4+3]*stride)/scale)
                val landmarks=FloatArray(10) { j ->
                    ((if (j%2==0) cx else cy)+points[anchor*10+j]*stride)/scale }
                candidates+=Detection(box,landmarks,scores[anchor])
            }
        }
        return nms(candidates,nms)
    }
}
