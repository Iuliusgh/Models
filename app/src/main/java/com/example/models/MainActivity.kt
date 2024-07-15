package com.example.models

import android.Manifest
import android.annotation.SuppressLint
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.RectF
import android.os.Bundle
import android.util.Size
import android.view.View
import android.view.ViewGroup
import android.widget.LinearLayout
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.constraintlayout.widget.ConstraintLayout
import androidx.constraintlayout.widget.ConstraintSet
import androidx.core.content.ContextCompat
import androidx.lifecycle.LifecycleOwner
import com.example.models.ui.theme.ObjectDetector
import com.google.android.gms.tflite.java.TfLite
import com.google.common.util.concurrent.ListenableFuture
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.InterpreterApi.Options.TfLiteRuntime
import org.tensorflow.lite.nnapi.NnApiDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import org.tensorflow.lite.support.image.ops.Rot90Op
import org.tensorflow.lite.task.vision.detector.Detection


class MainActivity : AppCompatActivity() {
    private lateinit var fpsView: TextView
    private lateinit var linearLayout: LinearLayout
    private lateinit var boxPrediction: View
    private lateinit var layout:ConstraintLayout
    private lateinit var parentLayout:ConstraintLayout
    private var tik = System.currentTimeMillis()
    private val inputSize:Int=160
    private lateinit var bitmapBuffer: Bitmap
    private var imageRotationDegrees: Int = 0
    private lateinit var previewView : PreviewView
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private val modelPath = "yolov8n_int8.tflite"
    private val labelPath = "yolov8n_int8_labels.txt"

    private val tfImageProcessor by lazy {
        val cropSize = minOf(inputSize, inputSize)
        ImageProcessor.Builder()
            .add(ResizeWithCropOrPadOp(cropSize, cropSize))
            .add(ResizeOp(
                tfInputSize.height, tfInputSize.width, ResizeOp.ResizeMethod.NEAREST_NEIGHBOR))
            .add(Rot90Op(-imageRotationDegrees / 90))
            .add(NormalizeOp(0f, 128f))
            .add(CastOp(DataType.FLOAT32))
            .build()
    }
    private val nnApiDelegate by lazy  {
        NnApiDelegate()
    }
    private val tflite by lazy {
        Interpreter(
            FileUtil.loadMappedFile(this, modelPath),
            Interpreter.Options().setRuntime(TfLiteRuntime.FROM_SYSTEM_ONLY).addDelegate(nnApiDelegate))
    }
    private val detector by lazy {
        ObjectDetector(
            tflite,
            FileUtil.loadLabels(this, labelPath)
        )
    }
    private val tfInputSize by lazy {
        val inputIndex = 0
        val inputShape = tflite.getInputTensor(inputIndex).shape()
        Size(inputShape[2], inputShape[1]) // Order of axis is: {1, height, width, 3}
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.camera_fragment)
        fpsView = findViewById(R.id.fpsView)
        parentLayout = findViewById(R.id.constraintLayout)
        // Start a timer or use a frame callback to update FPS
        TfLite.initialize(this)
        //get camera permissions
        if(checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            requestPermissions(arrayOf(Manifest.permission.CAMERA), 1)
        }
        //start camera
        previewView = findViewById(R.id.previewView)
        cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            bindCameraUseCase(cameraProvider)
        }, ContextCompat.getMainExecutor(this))

    }
    private fun computeFPS(tok:Long){
        val fps = 1000f/(tok - tik)
        val fpsText = "FPS: ${"%.2f".format(fps)}"
        fpsView.text = fpsText
        tik=System.currentTimeMillis()
    }
    public override fun onDestroy() {
        tflite.close()
        nnApiDelegate.close()
        super.onDestroy()

    }
    private fun bindCameraUseCase(cameraProvider : ProcessCameraProvider) {
        val preview : Preview = Preview.Builder()
            .build()

        val cameraSelector : CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(previewView.getSurfaceProvider())

        val imageAnalysis = ImageAnalysis.Builder()
            // enable the following line if RGBA output is needed.
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_BLOCK_PRODUCER)
            .build()

        imageAnalysis.setAnalyzer(ContextCompat.getMainExecutor(this)) { imageProxy ->
            if (!::bitmapBuffer.isInitialized) {
                // The image rotation and RGB image buffer are initialized only once
                // the analyzer has started running
                imageRotationDegrees = imageProxy.imageInfo.rotationDegrees
                bitmapBuffer = Bitmap.createBitmap(
                    imageProxy.width, imageProxy.height, Bitmap.Config.ARGB_8888
                )
            }
            bitmapBuffer.copyPixelsFromBuffer(imageProxy.planes[0].buffer)
            imageProxy.close()
            var tensorImageBuffer = TensorImage(DataType.FLOAT32)
            tensorImageBuffer.load(bitmapBuffer)
            var detections = detector.detect(tfImageProcessor.process(tensorImageBuffer))
            displayDetection(detections[0])

            // after done, release the ImageProxy object


            computeFPS(System.currentTimeMillis())
        }
        var camera = cameraProvider.bindToLifecycle(this as LifecycleOwner, cameraSelector, preview, imageAnalysis)
    }

    @SuppressLint("UseCompatLoadingForDrawables")
    private fun displayDetection(results: ObjectDetector.Detection?){
        layout=findViewById(R.id.resultLayout)
        layout.removeAllViews()
        val text = TextView(this)
        text.id = View.generateViewId()
        val boxPrediction = View(this)
        boxPrediction.id = View.generateViewId()
        boxPrediction.background=getDrawable(R.drawable.rectangle)
        text.layoutParams = ConstraintLayout.LayoutParams(ConstraintLayout.LayoutParams.WRAP_CONTENT,ConstraintLayout.LayoutParams.WRAP_CONTENT)
        boxPrediction.layoutParams= ConstraintLayout.LayoutParams(ConstraintLayout.LayoutParams.WRAP_CONTENT,ConstraintLayout.LayoutParams.WRAP_CONTENT)
        val constraintSet = ConstraintSet()

        val detectionResult = (results?.label ?: "") + " : " + (results?.score ?: "")
        text.text=detectionResult
        if (results != null) {
            val location = mapBoxCoords(results.location)
            (boxPrediction.layoutParams as ViewGroup.MarginLayoutParams).apply {
                leftMargin = location.left.toInt()
                topMargin = location.top.toInt()
                rightMargin = location.right.toInt()
                bottomMargin = location.bottom.toInt()
                width = (location.right - location.left).toInt()
                height = (location.bottom - location.top).toInt()
            }

            text.visibility=View.VISIBLE
            boxPrediction.visibility=View.VISIBLE
            layout.addView(text)
            layout.addView(boxPrediction)

            constraintSet.clone(layout)
            constraintSet.connect(text.id, ConstraintSet.BOTTOM,boxPrediction.id, ConstraintSet.TOP,10)
            constraintSet.connect(text.id, ConstraintSet.START,boxPrediction.id, ConstraintSet.START)
            constraintSet.connect(text.id, ConstraintSet.END,boxPrediction.id, ConstraintSet.END)
            constraintSet.connect(boxPrediction.id, ConstraintSet.TOP,text.id, ConstraintSet.BOTTOM)

            constraintSet.applyTo(layout)
            layout.invalidate()
        }

    }
    private fun mapBoxCoords(box: RectF):RectF{
        val boxLocation = RectF(
            box.left * previewView.width,
            box.top * previewView.height,
            box.right * previewView.width,
            box.bottom * previewView.height
        )
        val margin = 0.1f
        val ratio = previewView.width.toFloat() / previewView.height.toFloat()
        val midX = (boxLocation.left + boxLocation.right)/2f
        val midY = (boxLocation.top + boxLocation.bottom)/2f
        return if (previewView.width < previewView.height){
            RectF(
                midX - (1f + margin) * ratio * boxLocation.width() / 2f,
                midY - (1f - margin) * boxLocation.height() / 2f,
                midX + (1f + margin) * ratio * boxLocation.width() / 2f,
                midY + (1f - margin) * boxLocation.height() / 2f
            )
        } else {
            RectF(
                midX - (1f - margin) * boxLocation.width() / 2f,
                midY - (1f + margin) * ratio * boxLocation.height() / 2f,
                midX + (1f - margin) * boxLocation.width() / 2f,
                midY + (1f + margin) * ratio * boxLocation.height() / 2f
            )
        }

        }

}


