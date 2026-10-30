package com.nikitiva.sokolov

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import org.json.JSONObject
import java.io.File
import java.nio.FloatBuffer
import kotlin.math.min

/** All calls are serialized by MainActivity's analysis executor. */
class Engine(private val folder: File, private val manifest: JSONObject, val modelId: String): AutoCloseable {
    private val environment=OrtEnvironment.getEnvironment()
    private val detectorOptions=options()
    private val classifierOptions=options()
    private val detector=environment.createSession(File(folder,"detector.onnx").absolutePath,detectorOptions)
    private val modelConfig=(0 until manifest.getJSONArray("models").length()).map {
        manifest.getJSONArray("models").getJSONObject(it) }.first { it.getString("id")==modelId }
    private val classifier=environment.createSession(File(folder,modelConfig.getString("file")).absolutePath,classifierOptions)
    private val detectorConfig=manifest.getJSONObject("detector")
    private val size=modelConfig.getInt("input_size")
    private val input=FloatBuffer.allocate(3*size*size)
    private val detectorSize=detectorConfig.getInt("input_size")
    private val detectorBuffer=FloatBuffer.allocate(3*detectorSize*detectorSize)
    val technical=manifest.getBoolean("technical_test")
    data class FaceResult(val box: FloatArray, val score: Float?, val positive: Boolean, val error: String?=null)
    data class Result(val faces: List<FaceResult>, val detectionMs: Double, val classificationMs: Double,
                      val totalMs: Double, val model: String, val technical: Boolean)
    companion object {
        fun options() = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(1)
            addXnnpack(mapOf("intra_op_num_threads" to "4"))
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        fun validateSessions(folder: File) {
            val env=OrtEnvironment.getEnvironment()
            for(name in listOf("detector","resnet18","arcface")) {
                options().use { opts -> env.createSession(File(folder,"$name.onnx").absolutePath,opts).use { } }
            }
        }
    }
    private fun pixels(bitmap: Bitmap, buffer: FloatBuffer, normalizedDetector: Boolean) {
        val colors=IntArray(bitmap.width*bitmap.height)
        bitmap.getPixels(colors,0,bitmap.width,0,0,bitmap.width,bitmap.height)
        buffer.clear()
        for(channel in 0..2) for(color in colors) {
            val value=when(channel) { 0->Color.red(color); 1->Color.green(color); else->Color.blue(color) }
            buffer.put(if(normalizedDetector)(value-127.5f)/128f else value/255f)
        }
        buffer.rewind()
    }
    fun analyze(bitmap: Bitmap): Result {
        val start=System.nanoTime()
        val scale=min(detectorSize.toFloat()/bitmap.width,detectorSize.toFloat()/bitmap.height)
        val width=(bitmap.width*scale).toInt().coerceAtLeast(1)
        val height=(bitmap.height*scale).toInt().coerceAtLeast(1)
        // SCRFD pads bottom/right with black, preserving the upstream top-left origin.
        val resized=Bitmap.createScaledBitmap(bitmap,width,height,true)
        val padded=Bitmap.createBitmap(detectorSize,detectorSize,Bitmap.Config.ARGB_8888)
        Canvas(padded).drawBitmap(resized,0f,0f,null)
        pixels(padded,detectorBuffer,true)
        val outputs=OnnxTensor.createTensor(environment,detectorBuffer,
            longArrayOf(1,3,detectorSize.toLong(),detectorSize.toLong())).use { tensor ->
            detector.run(mapOf(detector.inputNames.first() to tensor)).use { results ->
                (0 until results.size()).map { i ->
                    val data=(results[i] as OnnxTensor).floatBuffer
                    FloatArray(data.remaining()).also { data.get(it) }
                }
            }
        }
        if(resized!==bitmap)resized.recycle(); padded.recycle()
        val detected=FaceMath.decode(outputs,detectorSize,height.toFloat()/bitmap.height,
            detectorConfig.getDouble("threshold").toFloat(),detectorConfig.getDouble("nms_threshold").toFloat())
        val detectedAt=System.nanoTime()
        val faces=detected.map { face ->
            try {
                val transform=FaceMath.similarity(face.landmarks)
                val matrix=Matrix().apply { setValues(floatArrayOf(transform[0].toFloat(),-transform[1].toFloat(),
                    transform[2].toFloat(),transform[1].toFloat(),transform[0].toFloat(),transform[3].toFloat(),0f,0f,1f)) }
                val aligned=Bitmap.createBitmap(224,224,Bitmap.Config.ARGB_8888)
                Canvas(aligned).drawBitmap(bitmap,matrix,Paint(Paint.FILTER_BITMAP_FLAG))
                val crop=if(size==224)aligned else Bitmap.createScaledBitmap(aligned,size,size,true)
                pixels(crop,input,false)
                val score=OnnxTensor.createTensor(environment,input,longArrayOf(1,3,size.toLong(),size.toLong())).use { tensor ->
                    classifier.run(mapOf("pixels" to tensor)).use { result ->
                        (result[0] as OnnxTensor).floatBuffer.get()
                    }
                }
                if(crop!==aligned)crop.recycle(); aligned.recycle()
                require(score.isFinite() && score in 0f..1f) { "Некорректная оценка модели" }
                FaceResult(face.box,score,score>=modelConfig.getDouble("threshold"))
            } catch(error: Exception) { FaceResult(face.box,null,false,"Не удалось проверить лицо") }
        }
        val end=System.nanoTime()
        return Result(faces,(detectedAt-start)/1e6,(end-detectedAt)/1e6,(end-start)/1e6,modelId,technical)
    }
    override fun close() { classifier.close(); detector.close(); classifierOptions.close(); detectorOptions.close() }
}
