package com.nikitiva.sokolov

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.Matrix
import android.net.Uri
import android.os.Bundle
import android.os.SystemClock
import android.util.Size
import android.view.View
import android.widget.Button
import android.widget.LinearLayout
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.exifinterface.media.ExifInterface
import java.util.concurrent.Executors
import java.util.concurrent.atomic.AtomicLong

class MainActivity: ComponentActivity() {
    private val worker=Executors.newSingleThreadExecutor()
    private val epoch=AtomicLong(0)
    private lateinit var store: ModelStore
    private var engine: Engine?=null // accessed only on worker
    private var provider: ProcessCameraProvider?=null
    private var analysis: ImageAnalysis?=null
    private lateinit var picture: ResultView
    private lateinit var status: TextView
    private lateinit var modelButton: Button
    private var modelId="resnet18"
    private var ready=false
    private var lastStarted=0L
    private var lastDisplayed=0L
    @Volatile private var cameraActive=false
    private val cameraPermission=registerForActivityResult(ActivityResultContracts.RequestPermission()) { allowed ->
        if(allowed)bindCamera() else message("Разрешите доступ к камере в настройках или выберите фото.")
    }
    private val photoPicker=registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        if(uri!=null)analyzePhoto(uri)
    }
    private val bundlePicker=registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if(uri!=null)importModels(uri)
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        store=ModelStore(this)
        modelId=getPreferences(0).getString("model","resnet18") ?: "resnet18"
        val root=LinearLayout(this).apply {
            orientation=LinearLayout.VERTICAL; setPadding(20,36,20,20); setBackgroundColor(Color.rgb(244,246,250))
        }
        root.addView(TextView(this).apply { text="Классификатор Соколова"; textSize=24f; setTextColor(Color.rgb(20,30,45)) })
        root.addView(TextView(this).apply { text="Камера и фотографии · обработка на телефоне"; textSize=14f })
        status=TextView(this).apply { textSize=14f; setPadding(0,12,0,12); accessibilityLiveRegion=View.ACCESSIBILITY_LIVE_REGION_POLITE }
        root.addView(status)
        picture=ResultView(this)
        root.addView(picture,LinearLayout.LayoutParams(-1,0,1f))
        val buttons=LinearLayout(this)
        buttons.addView(Button(this).apply { text="Камера"; setOnClickListener { startCamera() } },LinearLayout.LayoutParams(0,-2,1f))
        buttons.addView(Button(this).apply { text="Фото"; setOnClickListener {
            if(!ready){message("Сначала загрузите модели.");return@setOnClickListener}
            stopCamera(); photoPicker.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        } },LinearLayout.LayoutParams(0,-2,1f))
        root.addView(buttons)
        modelButton=Button(this).apply { text="Модель: $modelId"; setOnClickListener {
            stopCamera(); modelId=if(modelId=="resnet18")"arcface" else "resnet18"
            getPreferences(0).edit().putString("model",modelId).apply(); loadModel()
        } }
        root.addView(modelButton)
        root.addView(Button(this).apply { text="Загрузить модели (.zip)"; setOnClickListener {
            stopCamera(); bundlePicker.launch(arrayOf("application/zip","application/octet-stream","application/x-zip-compressed"))
        } })
        setContentView(root)
        loadModel()
    }
    private fun message(text: String) { if(!isDestroyed)status.text=text }
    private fun stopCamera() {
        epoch.incrementAndGet(); cameraActive=false; analysis?.clearAnalyzer(); provider?.unbindAll()
        lastDisplayed=0
    }
    private fun loadModel() {
        ready=false; modelButton.text="Модель: $modelId"; message("Открываю модель…")
        val token=epoch.incrementAndGet()
        worker.execute {
            try {
                engine?.close(); engine=null
                val manifest=store.manifest() ?: error("Сначала обучите модели в Kaggle и загрузите файл mobile_models.zip.")
                engine=Engine(store.folder,manifest,modelId)
                runOnUiThread { if(epoch.get()==token){ready=true; message(
                    if(engine!!.technical)"ТЕХНИЧЕСКИЕ ВЕСА — распознавание личности недостоверно."
                    else "Готово. Наведите камеру или выберите фотографию.")} }
            } catch(error: Exception) { runOnUiThread { if(epoch.get()==token)message(error.message ?: "Не удалось загрузить модель") } }
        }
    }
    private fun importModels(uri: Uri) {
        stopCamera(); ready=false; message("Проверяю и устанавливаю модели…")
        worker.execute {
            try {
                engine?.close(); engine=null; store.importBundle(uri)
                runOnUiThread { if(!isDestroyed)loadModel() }
            } catch(error: Exception) {
                engine=runCatching { store.manifest()?.let { Engine(store.folder,it,modelId) } }.getOrNull()
                val recovered=engine!=null
                runOnUiThread { ready=recovered
                    message("Не удалось установить модели: ${error.message}. " +
                        if(recovered)"Предыдущий комплект восстановлен." else "Выберите корректный комплект.") }
            }
        }
    }
    private fun startCamera() {
        if(!ready){message("Сначала загрузите модели.");return}
        if(ContextCompat.checkSelfPermission(this,Manifest.permission.CAMERA)==PackageManager.PERMISSION_GRANTED)bindCamera()
        else cameraPermission.launch(Manifest.permission.CAMERA)
    }
    private fun bindCamera() {
        stopCamera(); cameraActive=true
        val token=epoch.get()
        val future=ProcessCameraProvider.getInstance(this)
        future.addListener({
            if(!cameraActive || epoch.get()!=token)return@addListener
            try {
                provider=future.get()
                val useCase=ImageAnalysis.Builder().setTargetResolution(Size(640,480))
                    .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST).build()
                analysis=useCase
                useCase.setAnalyzer(worker) { proxy ->
                    try {
                        val now=SystemClock.elapsedRealtime()
                        if(!cameraActive || token!=epoch.get() || now-lastStarted<100)return@setAnalyzer
                        lastStarted=now
                        val original=proxy.toBitmap()
                        val rotation=proxy.imageInfo.rotationDegrees
                        val bitmap=if(rotation==0)original else Bitmap.createBitmap(original,0,0,original.width,original.height,
                            Matrix().apply { postRotate(rotation.toFloat()) },true).also { if(it!==original)original.recycle() }
                        process(bitmap,token)
                    } catch(error: Exception) {
                        runOnUiThread { if(epoch.get()==token)message("Ошибка камеры: ${error.message}") }
                    } finally { proxy.close() }
                }
                provider!!.bindToLifecycle(this,CameraSelector.DEFAULT_BACK_CAMERA,useCase)
                message("Ищу лица…")
            } catch(error: Exception) { message("Не удалось открыть камеру: ${error.message}") }
        },ContextCompat.getMainExecutor(this))
    }
    private fun analyzePhoto(uri: Uri) {
        stopCamera(); val token=epoch.get(); message("Проверяю фотографию…")
        worker.execute {
            try {
                val bounds=BitmapFactory.Options().apply { inJustDecodeBounds=true }
                contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it,null,bounds) }
                require(bounds.outWidth>0 && bounds.outHeight>0) { "Не удалось прочитать изображение" }
                var sample=1
                while(bounds.outWidth/sample>2048 || bounds.outHeight/sample>2048)sample*=2
                val bitmap=contentResolver.openInputStream(uri)!!.use { BitmapFactory.decodeStream(it,null,
                    BitmapFactory.Options().apply { inSampleSize=sample; inPreferredConfig=Bitmap.Config.ARGB_8888 }) }
                    ?: error("Не удалось прочитать фото")
                val orientation=contentResolver.openInputStream(uri)!!.use { ExifInterface(it).getAttributeInt(
                    ExifInterface.TAG_ORIENTATION,ExifInterface.ORIENTATION_NORMAL) }
                val matrix=Matrix()
                when(orientation) {
                    ExifInterface.ORIENTATION_ROTATE_90->matrix.postRotate(90f)
                    ExifInterface.ORIENTATION_ROTATE_180->matrix.postRotate(180f)
                    ExifInterface.ORIENTATION_ROTATE_270->matrix.postRotate(270f)
                    ExifInterface.ORIENTATION_FLIP_HORIZONTAL->matrix.postScale(-1f,1f)
                    ExifInterface.ORIENTATION_FLIP_VERTICAL->matrix.postScale(1f,-1f)
                    ExifInterface.ORIENTATION_TRANSPOSE->{matrix.postScale(-1f,1f);matrix.postRotate(270f)}
                    ExifInterface.ORIENTATION_TRANSVERSE->{matrix.postScale(-1f,1f);matrix.postRotate(90f)}
                }
                val oriented=if(matrix.isIdentity)bitmap else Bitmap.createBitmap(bitmap,0,0,bitmap.width,bitmap.height,matrix,true)
                    .also { if(it!==bitmap)bitmap.recycle() }
                process(oriented,token)
            } catch(error: Exception) {runOnUiThread {if(epoch.get()==token)message("Ошибка фото: ${error.message}")} }
        }
    }
    private fun process(bitmap: Bitmap,token: Long) {
        try {
            val result=(engine ?: error("Модель не загружена")).analyze(bitmap)
            runOnUiThread {
                if(isDestroyed || epoch.get()!=token){bitmap.recycle();return@runOnUiThread}
                picture.show(bitmap,result)
                val now=SystemClock.elapsedRealtime()
                val fps=if(lastDisplayed>0)1000.0/(now-lastDisplayed).coerceAtLeast(1) else 0.0
                lastDisplayed=now
                val verdict=when {
                    result.technical->"ТЕХНИЧЕСКИЕ ВЕСА — это не проверка личности"
                    result.faces.isEmpty()->"Лица не найдены — приблизьте камеру или выберите другое фото"
                    result.faces.any { it.positive }->"Найден Соколов"
                    result.faces.any { it.error!=null }->"Часть лиц не удалось проверить"
                    else->"Соколов не найден"
                }
                message("$verdict\nЛиц: ${result.faces.size} · ${result.model} · весь кадр %.0f мс\nПоиск %.0f мс · классификация %.0f мс%s".format(
                    result.totalMs,result.detectionMs,result.classificationMs,if(cameraActive)" · %.1f проверок/с".format(fps) else ""))
            }
        } catch(error: Exception) { bitmap.recycle(); throw error }
    }
    override fun onPause() { stopCamera(); super.onPause() }
    override fun onDestroy() {
        worker.execute { engine?.close();engine=null }; worker.shutdown(); super.onDestroy()
    }
}
