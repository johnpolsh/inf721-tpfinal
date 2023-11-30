package com.jopzoli.objectdetection

import android.annotation.SuppressLint
import android.graphics.Bitmap
import android.os.Bundle
import android.os.SystemClock
import android.speech.tts.TextToSpeech
import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.Toast
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.fragment.app.Fragment
import androidx.navigation.Navigation
import com.jopzoli.objectdetection.databinding.FragmentCameraBinding
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors


class ObjectDetector(
    private val module: Module,
    private val listener: Listener
) {
    private val _cTAG0 = "ObjectDetection.ObjectDetector"

    private var lastTime: Long = 0

    private fun preProcess(bitmap: Bitmap, imageRotation: Int): Bitmap {
        return Bitmap.createScaledBitmap(
            bitmap,
            PrePostProcessor.INPUT_WIDTH,
            PrePostProcessor.INPUT_HEIGHT,
            true
        )
    }

    fun detect(image: Bitmap, imageRotation: Int) {
        if (SystemClock.uptimeMillis() - lastTime < 3000)
            return
        lastTime = SystemClock.uptimeMillis()
        var inferenceTime = SystemClock.uptimeMillis()
        val inputBitmap = preProcess(image, imageRotation)
        val input = TensorImageUtils.bitmapToFloat32Tensor(
            inputBitmap,
            PrePostProcessor.NO_MEAN_RGB,
            PrePostProcessor.NO_STD_RGB
        )

        val output = module.forward(IValue.from(input)).toTuple()[0].toTensor()
        Log.e(_cTAG0, output.dataAsFloatArray.toString())

        val results = listener.let {
            PrePostProcessor.outputsToNMSPredictions(
                output.dataAsFloatArray,
                it.imgScaleX,
                listener.imgScaleY,
                listener.ivScaleX,
                listener.ivScaleY,
                listener.startX,
                listener.startY
            )
        }
        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        listener.onDetection(
            results,
            inferenceTime
        )
    }

    interface Listener {
        var imgScaleX: Float
        var imgScaleY: Float
        var ivScaleX: Float
        var ivScaleY: Float
        var startX: Float
        var startY: Float

        fun onDetection(
            results: ArrayList<Detection>?,
            inferenceTime: Long
        )

        fun onError(error: String)
    }
}

class CameraFragment : Fragment(), ObjectDetector.Listener {
    private val _cTAG0 = "ObjectDetection.CameraFragment"

    private var _fragmentCameraBinding: FragmentCameraBinding? = null

    private val fragmentCameraBinding
        get() = _fragmentCameraBinding!!

    private lateinit var objectDetector: ObjectDetector
    private lateinit var bitmapBuffer: Bitmap
    private var preview: Preview? = null
    private var imageAnalyzer: ImageAnalysis? = null
    private var camera: Camera? = null
    private var cameraProvider: ProcessCameraProvider? = null
    private var lastDetection: Int = 0
    private var lastDetectionCount: Int = 0

    private lateinit var cameraExecutor: ExecutorService

    private lateinit var tts: TextToSpeech

    override var imgScaleX = 1f
    override var imgScaleY = 1f
    override var ivScaleX = 1f
    override var ivScaleY = 1f
    override var startX = 0f
    override var startY = 0f

    private fun detectObjects(image: ImageProxy) {
        image.use { bitmapBuffer.copyPixelsFromBuffer(image.planes[0].buffer) }

        val imageRotation = image.imageInfo.rotationDegrees
        objectDetector.detect(bitmapBuffer, imageRotation)
    }

    private fun setUpCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(requireContext())
        cameraProviderFuture.addListener(
            {
                cameraProvider = cameraProviderFuture.get()
                bindCameraUseCases()
            },
            ContextCompat.getMainExecutor(requireContext())
        )
    }

    @SuppressLint("UnsafeOptInUsageError")
    private fun bindCameraUseCases() {
        val cameraProvider =
            cameraProvider ?: throw IllegalStateException("Camera initialization failed.")
        val cameraSelector =
            CameraSelector.Builder().requireLensFacing(CameraSelector.LENS_FACING_BACK).build()

        val resolutionSelector = ResolutionSelector.Builder()
            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
            .build()

        preview =
            Preview.Builder()
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .build()


        imageAnalyzer =
            ImageAnalysis.Builder()
                .setResolutionSelector(resolutionSelector)
                .setTargetRotation(fragmentCameraBinding.viewFinder.display.rotation)
                .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                .setOutputImageFormat(OUTPUT_IMAGE_FORMAT_RGBA_8888)
                .build()
                .also {
                    it.setAnalyzer(cameraExecutor) { image ->
                        imgScaleX = 1f
                        imgScaleY = 1f
                        ivScaleX = 1f
                        ivScaleY = 1f
                        startX = 0f
                        startY = 0f

                        if (!::bitmapBuffer.isInitialized) {
                            bitmapBuffer = Bitmap.createBitmap(
                                image.width,
                                image.height,
                                Bitmap.Config.ARGB_8888
                            )
                            imgScaleX = (image.width / PrePostProcessor.INPUT_WIDTH).toFloat()
                            imgScaleY = (image.height / PrePostProcessor.INPUT_HEIGHT).toFloat()
                            ivScaleX = if (image.width > image.height)
                                (fragmentCameraBinding.overlay.width / image.width).toFloat()
                            else
                                (fragmentCameraBinding.overlay.height / image.height).toFloat()
                            ivScaleY = if (image.height > image.width)
                                (fragmentCameraBinding.overlay.height / image.height).toFloat()
                            else
                                (fragmentCameraBinding.overlay.width / image.width).toFloat()
                            startX =
                                (fragmentCameraBinding.overlay.width - ivScaleX * image.width) / 2f
                            startY =
                                (fragmentCameraBinding.overlay.height - ivScaleY * image.height) / 2f
                        }

                        detectObjects(image)
                    }
                }

        cameraProvider.unbindAll()

        try {
            camera = cameraProvider.bindToLifecycle(
                this,
                cameraSelector,
                preview,
                imageAnalyzer
            )

            preview?.setSurfaceProvider(fragmentCameraBinding.viewFinder.surfaceProvider)
        } catch (exc: Exception) {
            Log.e(_cTAG0, "Use case binding failed", exc)
        }
    }

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _fragmentCameraBinding = FragmentCameraBinding.inflate(inflater, container, false)
        tts = TextToSpeech(context) { status ->
            if (status == TextToSpeech.SUCCESS) {
                val result = tts.setLanguage(Locale.getDefault())
                if (result == TextToSpeech.LANG_MISSING_DATA ||
                    result == TextToSpeech.LANG_NOT_SUPPORTED
                ) {
                    Log.e(_cTAG0, "This Language is not supported")
                }
            } else Log.e(_cTAG0, "TTS initialization Failed")
        }

        return fragmentCameraBinding.root
    }

    override fun onDestroyView() {
        _fragmentCameraBinding = null
        super.onDestroyView()
        cameraExecutor.shutdown()
    }

    override fun onDetection(
        results: ArrayList<Detection>?,
        inferenceTime: Long
    ) {
        results?.sortBy {
            it.score
        }
        activity?.runOnUiThread {
            fragmentCameraBinding.overlay.setDetectionResults(ArrayList())
            fragmentCameraBinding.overlay.invalidate()
//            fragmentCameraBinding.inferenceTime.setText
            if (results != null) {
                if (results.size != 0) {
                    var speak = MainActivity.classes[results[0].classIdx]
                    if (results[0].classIdx == 0 && (0 .. 4).random() <= 1)
                        speak += " bonita"
                    tts.speak(
                        MainActivity.classes[results[0].classIdx],
                        TextToSpeech.QUEUE_ADD,
                        null
                    )
                }
            }
        }
    }

    override fun onError(error: String) {
        activity?.runOnUiThread {
            Log.e(_cTAG0, error)
            Toast.makeText(requireContext(), error, Toast.LENGTH_SHORT).show()
        }
    }

    override fun onResume() {
        super.onResume()
        if (!PermissionsFragment.hasPermissions(requireContext())) {
            Navigation.findNavController(requireActivity(), R.id.fragment_container)
                .navigate(R.id.permissions_fragment)
        }
    }

    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
        super.onViewCreated(view, savedInstanceState)
        objectDetector = ObjectDetector(MainActivity.module, this)

        cameraExecutor = Executors.newSingleThreadExecutor()
        fragmentCameraBinding.viewFinder.post {
            setUpCamera()
        }
    }
}