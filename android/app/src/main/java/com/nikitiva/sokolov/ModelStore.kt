package com.nikitiva.sokolov

import android.content.Context
import android.net.Uri
import org.json.JSONObject
import java.io.File
import java.security.MessageDigest
import java.util.zip.ZipInputStream

/** Only flat known files are accepted; imports are staged before activation. */
class ModelStore(private val context: Context) {
    val folder=File(context.filesDir,"models")
    private val allowed=setOf("manifest.json","detector.onnx","resnet18.onnx","arcface.onnx")
    fun sha256(file: File): String {
        val digest=MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer=ByteArray(65536)
            while (true) { val n=input.read(buffer); if(n<0) break; digest.update(buffer,0,n) }
        }
        return digest.digest().joinToString("") { "%02x".format(it.toInt() and 255) }
    }
    fun manifest(): JSONObject? = if(File(folder,"manifest.json").exists()) validate(folder) else null
    fun validate(directory: File): JSONObject {
        val manifestFile=File(directory,"manifest.json")
        require(manifestFile.length() in 1..65536) { "Некорректный manifest.json" }
        val manifest=JSONObject(manifestFile.readText())
        require(manifest.getInt("schema_version")==1)
        manifest.getBoolean("technical_test")
        require(manifest.getJSONArray("class_names").toString()=="[\"not_sokolov\",\"sokolov\"]")
        require(manifest.getString("default_model")=="resnet18")
        val models=manifest.getJSONArray("models")
        require(models.length()==2)
        val ids=mutableSetOf<String>()
        for(i in 0 until models.length()) {
            val model=models.getJSONObject(i); val id=model.getString("id")
            require(id in setOf("resnet18","arcface") && ids.add(id))
            require(model.getString("file")=="$id.onnx")
            require(model.getInt("input_size")==if(id=="resnet18")224 else 112)
            require(model.getString("input_name")=="pixels" && model.getString("output_name")=="score")
            require(model.getDouble("threshold").isFinite() && model.getDouble("threshold") in 0.0..1.000001)
            require(sha256(File(directory,"$id.onnx"))==model.getString("sha256")) { "Повреждена модель $id" }
        }
        val detector=manifest.getJSONObject("detector")
        require(detector.getString("file")=="detector.onnx" && detector.getInt("input_size")==640)
        require(detector.getDouble("threshold") in 0.0..1.0 && detector.getDouble("nms_threshold") in 0.0..1.0)
        require(sha256(File(directory,"detector.onnx"))==detector.getString("sha256")) { "Повреждён детектор" }
        return manifest
    }
    fun importBundle(uri: Uri) {
        val staged=File(context.filesDir,"models-staged"); staged.deleteRecursively(); staged.mkdirs()
        try {
            var total=0L; val seen=mutableSetOf<String>()
            ZipInputStream(context.contentResolver.openInputStream(uri)!!).use { zip ->
                while(true) {
                    val entry=zip.nextEntry ?: break
                    require(!entry.isDirectory && entry.name in allowed && seen.add(entry.name)) { "Неизвестный файл в архиве" }
                    File(staged,entry.name).outputStream().use { out ->
                        val buffer=ByteArray(65536)
                        while(true) { val n=zip.read(buffer); if(n<0) break
                            total+=n; require(total<=400L*1024*1024) { "Архив больше 400 МБ" }; out.write(buffer,0,n) }
                    }
                    zip.closeEntry()
                }
            }
            require(seen==allowed) { "Нужны обе модели, детектор и manifest.json" }
            validate(staged)
            // Validate that ORT can load all three files before replacing a working bundle.
            Engine.validateSessions(staged)
            val previous=File(context.filesDir,"models-backup"); previous.deleteRecursively()
            if(folder.exists()) require(folder.renameTo(previous))
            if(!staged.renameTo(folder)) { previous.renameTo(folder); error("Не удалось сохранить модели") }
            previous.deleteRecursively()
        } finally { staged.deleteRecursively() }
    }
}
