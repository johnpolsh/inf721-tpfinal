package com.jopzoli.objectdetection

import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.appcompat.app.AppCompatActivity
import com.jopzoli.objectdetection.databinding.ActivityMainBinding
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import java.io.BufferedReader
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import java.io.InputStreamReader


class MainActivity : AppCompatActivity() {
    private val _cTAG0 = "ObjectDetection.MainActivity"

    private lateinit var binding: ActivityMainBinding

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        val modelFile = "yolov5s.torchscript.ptl"
        val classesFile = "classes-pt.txt"
        try {
            module = loadModule(this, modelFile)
        } catch (e: Exception) {
            Log.e(_cTAG0, "Could not load torch model from file $modelFile", e)
        }
        try {
            classes = loadClasses(this, classesFile)
            Log.i(_cTAG0, "Loaded classes: $classes")
        } catch (e: Exception) {
            Log.e(_cTAG0, "Could not load model classes from file $classesFile", e)
        }
    }

    companion object {
        lateinit var module: Module
        lateinit var classes: ArrayList<String>

        fun loadModule(context: Context, name: String): Module {
            val path = assetFilePath(context, name)

            return LiteModuleLoader.load(path)
        }

        fun loadClasses(context: Context, name: String): ArrayList<String> {
            val br = BufferedReader(InputStreamReader(context.assets.open(name)))
            val classes = ArrayList<String>()
            br.lineSequence().forEach { classes.add(it) }

            return classes
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