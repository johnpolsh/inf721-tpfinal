package com.jopzoli.objectdetection

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.jopzoli.objectdetection.databinding.ActivityMainBinding
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

class MainActivity : AppCompatActivity() {
    private val _cTAG0 = "ObjectDetection.MainActivity"

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val modelName = "yolov5s.torchscript.ptl"
        try {
            module = loadModule(this, modelName)
        } catch (e: Exception) {
            Log.e(_cTAG0, "Could not load torch model from file $modelName", e)
        }
    }

    companion object {
        lateinit var module: Module

        fun loadModule(context: Context, name: String): Module {
            val path = assetFilePath(context, name)

            return LiteModuleLoader.load(path)
        }

        @Throws(IOException::class)
        fun assetFilePath(context: Context, assetName: String): String? {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }
            context.assets.open(assetName).use { `is` ->
                FileOutputStream(file).use { os ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (`is`.read(buffer).also { read = it } != -1) {
                        os.write(buffer, 0, read)
                    }
                    os.flush()
                }
                return file.absolutePath
            }
        }
    }
}