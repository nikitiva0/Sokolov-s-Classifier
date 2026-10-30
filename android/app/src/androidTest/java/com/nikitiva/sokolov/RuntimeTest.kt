package com.nikitiva.sokolov

import android.Manifest
import android.app.Activity
import android.app.Instrumentation
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Color
import android.net.Uri
import android.os.SystemClock
import android.widget.TextView
import androidx.test.core.app.ActivityScenario
import androidx.test.espresso.Espresso.onView
import androidx.test.espresso.action.ViewActions.click
import androidx.test.espresso.intent.Intents
import androidx.test.espresso.intent.matcher.IntentMatchers.hasAction
import androidx.test.espresso.intent.Intents.intending
import androidx.test.espresso.matcher.ViewMatchers.withText
import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import androidx.test.rule.GrantPermissionRule
import org.hamcrest.Matchers.anyOf
import org.junit.Assert.*
import org.junit.Rule
import org.junit.Test
import org.junit.runner.RunWith
import java.io.File

@RunWith(AndroidJUnit4::class)
class RuntimeTest {
    @get:Rule val permission: GrantPermissionRule=GrantPermissionRule.grant(Manifest.permission.CAMERA)
    private val instrumentation=InstrumentationRegistry.getInstrumentation()
    private val context get()=instrumentation.targetContext
    private fun fixture(): Uri {
        val file=File(context.cacheDir,"TECHNICAL_FIXTURE.zip")
        instrumentation.context.assets.open("fixture_models.zip").use { input ->
            file.outputStream().use { output -> input.copyTo(output) } }
        return Uri.fromFile(file)
    }
    private fun status(activity: MainActivity): String {
        return activity.findViewById<TextView>(R.id.status).text.toString()
    }
    private fun waitStatus(scenario: ActivityScenario<MainActivity>,part: String): String {
        val deadline=SystemClock.elapsedRealtime()+20000
        var text=""
        while(SystemClock.elapsedRealtime()<deadline) {
            scenario.onActivity { text=status(it) }
            if(text.contains(part))return text
            SystemClock.sleep(100)
        }
        fail("Status did not contain '$part': $text")
        return text
    }
    @Test fun missingModelsExplainRequiredImportInsteadOfGuessing() {
        ModelStore(context).folder.deleteRecursively()
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            waitStatus(scenario,"Сначала обучите модели")
            onView(withText("Камера")).perform(click())
            waitStatus(scenario,"Сначала загрузите модели")
            onView(withText("Фото")).perform(click())
            waitStatus(scenario,"Сначала загрузите модели")
        }
    }
    @Test fun nativeOrtImportBothModelsNoFaceAndInvalidImportPreservesModels() {
        val store=ModelStore(context)
        store.folder.deleteRecursively()
        store.importBundle(fixture())
        val manifest=store.manifest()!!
        assertTrue(manifest.getBoolean("technical_test"))
        val white=Bitmap.createBitmap(640,640,Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        val black=Bitmap.createBitmap(640,640,Bitmap.Config.ARGB_8888).apply { eraseColor(Color.BLACK) }
        for(id in listOf("resnet18","arcface")) {
            Engine(store.folder,manifest,id).use { engine ->
                val result=engine.analyze(white)
                assertTrue(result.technical); assertEquals(1,result.faces.size)
                assertNull(result.faces.single().error)
                assertEquals(if(id=="resnet18").9 else .1,result.faces.single().score!!.toDouble(),1e-5)
                assertEquals(id=="resnet18",result.faces.single().positive)
                assertTrue(engine.analyze(black).faces.isEmpty())
            }
        }
        val before=store.sha256(File(store.folder,"manifest.json"))
        val corrupt=File(context.cacheDir,"corrupt.zip").apply { writeText("not a zip") }
        try { store.importBundle(Uri.fromFile(corrupt)); fail("Corrupt ZIP accepted") } catch(_: IllegalArgumentException) { }
        assertEquals(before,store.sha256(File(store.folder,"manifest.json")))
        white.recycle();black.recycle()
    }
    @Test fun galleryResultAndCameraFramesReachUiWithExplicitTechnicalWarning() {
        ModelStore(context).importBundle(fixture())
        // Set the app's persisted default, as a real first launch would do.
        context.getSharedPreferences("MainActivity",0).edit().putString("model","resnet18").commit()
        val photo=File(context.cacheDir,"fixture-photo.png")
        Bitmap.createBitmap(640,640,Bitmap.Config.ARGB_8888).apply {
            eraseColor(Color.WHITE); photo.outputStream().use { compress(Bitmap.CompressFormat.PNG,100,it) };recycle() }
        ActivityScenario.launch(MainActivity::class.java).use { scenario ->
            waitStatus(scenario,"ТЕХНИЧЕСКИЕ ВЕСА")
            Intents.init()
            try {
                val corrupt=File(context.cacheDir,"bad-models.zip").apply { writeText("not a zip") }
                intending(hasAction(Intent.ACTION_OPEN_DOCUMENT)).respondWith(
                    Instrumentation.ActivityResult(Activity.RESULT_OK,Intent().setData(Uri.fromFile(corrupt))))
                onView(withText("Загрузить модели (.zip)")).perform(click())
                waitStatus(scenario,"Предыдущий комплект восстановлен")
            } finally { Intents.release() }
            Intents.init()
            try {
                intending(anyOf(hasAction("android.provider.action.PICK_IMAGES"),
                    hasAction(Intent.ACTION_OPEN_DOCUMENT),hasAction(Intent.ACTION_GET_CONTENT)))
                    .respondWith(Instrumentation.ActivityResult(Activity.RESULT_OK,Intent().setData(Uri.fromFile(photo))))
                onView(withText("Фото")).perform(click())
                val text=waitStatus(scenario,"Лиц: 1")
                assertTrue(text.contains("ТЕХНИЧЕСКИЕ ВЕСА"))
            } finally { Intents.release() }
            onView(withText("Камера")).perform(click())
            val camera=waitStatus(scenario,"проверок/с")
            assertTrue(camera.contains("ТЕХНИЧЕСКИЕ ВЕСА"))
            assertFalse(camera.contains("Ошибка камеры"))
            instrumentation.uiAutomation.takeScreenshot()?.let { screenshot ->
                File(context.getExternalFilesDir(null),"runtime-camera.png").outputStream().use {
                    screenshot.compress(Bitmap.CompressFormat.PNG,100,it) }
                screenshot.recycle()
            }
            // Exercise a lifecycle restart: no stale model/bitmap or executor callback.
            scenario.recreate()
            waitStatus(scenario,"ТЕХНИЧЕСКИЕ ВЕСА")
        }
    }
}
