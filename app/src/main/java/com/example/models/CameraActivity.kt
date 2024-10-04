package com.example.models

/*

class CameraActivity: AppCompatActivity() {
    private lateinit var activityCameraBinding: ActivityCameraBinding
    private var tik = System.currentTimeMillis()
    private lateinit var bitmapBuffer: Bitmap
    private lateinit var previewView: PreviewView
    private lateinit var cameraProviderFuture: ListenableFuture<ProcessCameraProvider>
    private var imageRotationDegrees: Int = 0

    private fun onCreate() {
//get camera permissions

        if (checkSelfPermission(Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
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

    private fun computeFPS(tok: Long) {
        val fps = 1000f / (tok - tik)
        val fpsText = "FPS: ${"%.2f".format(fps)}"
        activityCameraBinding.fpsView.text = fpsText
        tik = System.currentTimeMillis()
    }

    private fun bindCameraUseCase(cameraProvider: ProcessCameraProvider) {
        val preview: Preview = Preview.Builder()
            .build()

        val cameraSelector: CameraSelector = CameraSelector.Builder()
            .requireLensFacing(CameraSelector.LENS_FACING_BACK)
            .build()

        preview.setSurfaceProvider(previewView.getSurfaceProvider())

        val imageAnalysis = ImageAnalysis.Builder()
// enable the following line if RGBA output is needed.
            .setOutputImageFormat(ImageAnalysis.OUTPUT_IMAGE_FORMAT_RGBA_8888)
            .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
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
            val tensorImageBuffer = TensorImage(DataType.FLOAT32)
            tensorImageBuffer.load(bitmapBuffer)
            //var detections = detector.detect(tfImageProcessor.process(tensorImageBuffer))
            //activityCameraBinding.overlayView.setResults(listOf(detections.maxBy { it.categories[0].score }))
            activityCameraBinding.overlayView.invalidate()
// after done, release the ImageProxy object

            computeFPS(System.currentTimeMillis())
        }
        var camera = cameraProvider.bindToLifecycle(
            this as LifecycleOwner,
            cameraSelector,
            preview,
            imageAnalysis
        )
    }

    @SuppressLint("UseCompatLoadingForDrawables")
    private fun displayDetection(results: Detection?) {

    }

    private fun mapBoxCoords(box: RectF): RectF {
        val boxLocation = RectF(
            box.left * previewView.width,
            box.top * previewView.height,
            box.right * previewView.width,
            box.bottom * previewView.height
        )
        val margin = 0.1f
        val ratio = previewView.width.toFloat() / previewView.height.toFloat()
        val midX = (boxLocation.left + boxLocation.right) / 2f
        val midY = (boxLocation.top + boxLocation.bottom) / 2f
        return if (previewView.width < previewView.height) {
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
}*/