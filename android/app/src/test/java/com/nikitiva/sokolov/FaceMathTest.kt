package com.nikitiva.sokolov

import org.junit.Assert.*
import org.junit.Test

class FaceMathTest {
    @Test fun detectorKeepsTightCropsAtNativeScale() {
        assertEquals(1f,FaceMath.detectorScale(224,224,640,true),0f)
        assertEquals(1f,FaceMath.detectorScale(240,116,640,true),0f)
        assertEquals(640f/1920,FaceMath.detectorScale(1920,1080,640,true),1e-6f)
        assertEquals(640f/224,FaceMath.detectorScale(224,224,640,false),1e-6f)
    }
    @Test fun similarityRecoversKnownTransform() {
        val template=FaceMath.template
        val input=FloatArray(10) { i -> ((template[i]-(if(i%2==0)10 else 20))/2).toFloat() }
        val matrix=FaceMath.similarity(input)
        assertEquals(2.0,matrix[0],1e-5); assertEquals(0.0,matrix[1],1e-5)
        assertEquals(10.0,matrix[2],1e-4); assertEquals(20.0,matrix[3],1e-4)
    }
    @Test fun nmsRemovesOverlappingLowerScore() {
        val a=FaceMath.Detection(floatArrayOf(0f,0f,100f,100f),FloatArray(10),.9f)
        val b=FaceMath.Detection(floatArrayOf(1f,1f,99f,99f),FloatArray(10),.8f)
        assertEquals(listOf(a),FaceMath.nms(listOf(b,a),.4f))
    }
    @Test fun scrfdDecodesAnchorsAndFiveLandmarks() {
        val size=32
        val outputs=(0..8).map { i->
            val stride=listOf(8,16,32)[i%3]
            FloatArray((size/stride)*(size/stride)*2*listOf(1,4,10)[i/3]) }
        outputs[0][2]=.9f // stride8, anchor index2 => center x8,y0
        for(j in 0..3)outputs[3][2*4+j]=1f
        val face=FaceMath.decode(outputs,size,1f,.5f,.4f).single()
        assertArrayEquals(floatArrayOf(0f,-8f,16f,8f),face.box,.001f)
        assertArrayEquals(floatArrayOf(8f,0f,8f,0f,8f,0f,8f,0f,8f,0f),face.landmarks,.001f)
    }
}
