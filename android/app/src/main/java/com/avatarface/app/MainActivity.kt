package com.avatarface.app

import ai.onnxruntime.OnnxTensor
import android.graphics.Bitmap
import com.avatarface.app.render.AttributeParser
import com.avatarface.app.render.AvatarAttributes
import com.avatarface.app.render.AvatarRenderer
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.app.Activity
import android.os.Bundle
import android.os.Debug
import android.os.SystemClock
import android.util.Log
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.util.Locale
import org.json.JSONArray
import org.json.JSONObject

class MainActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val backend = intent.getStringExtra("backend") ?: "cpu"
        val modelAsset = intent.getStringExtra("model") ?: DEFAULT_MODEL_ASSET
        val profileOperators = intent.getBooleanExtra("profile_operators", false)
        val runs = intent.getIntExtra("runs", DEFAULT_RUNS).coerceIn(1, MAX_RUNS)
        val mode = intent.getStringExtra("mode") ?: "benchmark"
        Thread {
            val payload = try {
                if (mode == "render") renderGallery(runs) else
                    runBenchmark(backend, modelAsset, runs, profileOperators)
            } catch (error: Throwable) {
                JSONObject()
                    .put("schema_version", 1)
                    .put("status", "error")
                    .put("backend", backend)
                    .put("backend_requested", backend)
                    .put("model", modelAsset)
                    .put("error_type", error.javaClass.name)
                    .put("error", error.message ?: "unknown")
            }
            val output = File(filesDir, RESULT_FILE)
            output.writeText(payload.toString(2))
            Log.i(TAG, "RESULT ${payload}")
            finish()
        }.start()
    }

    /**
     * Dibuja la galería del ADR 0012 con el trazado nativo y mide el tiempo.
     *
     * Las especificaciones vienen del asset `gallery-specs.json`, generado por
     * `scripts/render_gallery.py --dump-specs`, para que la app y Python
     * dibujen exactamente las mismas personas y sus salidas sean comparables.
     */
    private fun renderGallery(runs: Int): JSONObject {
        val specs = JSONArray(
            assets.open(GALLERY_SPECS).use { it.readBytes().toString(Charsets.UTF_8) },
        )
        val renderer = AvatarRenderer(AVATAR_SIZE)
        val directory = File(filesDir, "render").apply { mkdirs() }
        directory.listFiles()?.forEach { it.delete() }
        val durations = JSONArray()
        val identifiers = JSONArray()
        for (index in 0 until specs.length()) {
            val spec = specs.getJSONObject(index)
            val attributes = AvatarAttributes.fromJson(spec)
            // Calentamiento fuera de la medida: la primera llamada carga clases.
            renderer.render(attributes).recycle()
            var best = Double.MAX_VALUE
            var bitmap: Bitmap? = null
            repeat(runs) {
                val started = SystemClock.elapsedRealtimeNanos()
                val rendered = renderer.render(attributes)
                best = minOf(best, elapsedMilliseconds(started))
                bitmap?.recycle()
                bitmap = rendered
            }
            val identifier = spec.optString("identifier", "persona-%02d".format(index + 1))
            File(directory, "$identifier.png").outputStream().use { stream ->
                bitmap?.compress(Bitmap.CompressFormat.PNG, 100, stream)
            }
            bitmap?.recycle()
            durations.put(best)
            identifiers.put(identifier)
        }
        val sorted = (0 until durations.length()).map { durations.getDouble(it) }.sorted()
        return JSONObject()
            .put("parser_mismatches", verifyParser())
            .put("schema_version", 1)
            .put("status", "ok")
            .put("mode", "render")
            .put("avatars", specs.length())
            .put("image_size", AVATAR_SIZE)
            .put("runs_per_avatar", runs)
            .put("durations_ms", durations)
            .put("identifiers", identifiers)
            .put("median_ms", sorted[sorted.size / 2])
            .put("max_ms", sorted.last())
            .put("output_directory", directory.absolutePath)
    }

    /**
     * Comprueba que el parser de texto nativo coincide con el de Python.
     *
     * Los casos vienen del asset `parser-cases.json`, generado desde Python con
     * los atributos que espera; devolver una lista vacía significa que las dos
     * implementaciones interpretan igual las mismas frases.
     */
    private fun verifyParser(): JSONArray {
        val mismatches = JSONArray()
        val cases = JSONArray(
            assets.open(PARSER_CASES).use { it.readBytes().toString(Charsets.UTF_8) },
        )
        for (index in 0 until cases.length()) {
            val case = cases.getJSONObject(index)
            val text = case.getString("text")
            val expected = case.getJSONObject("expected")
            val actual = AttributeParser.parse(text)
            val fields = mapOf(
                "expression" to actual.expression,
                "face_shape" to actual.faceShape,
                "skin_tone" to actual.skinTone,
                "hair_style" to actual.hairStyle,
                "hair_color" to actual.hairColor,
                "eye_color" to actual.eyeColor,
                "eye_shape" to actual.eyeShape,
                "accessory" to actual.accessory,
                "background" to actual.background,
                "brow_style" to actual.browStyle,
                "nose_style" to actual.noseStyle,
                "facial_hair" to actual.facialHair,
                "glasses" to actual.glasses,
                "earrings" to actual.earrings,
                "freckles" to actual.freckles,
                "clothing" to actual.clothing,
                "clothing_color" to actual.clothingColor,
            )
            for ((key, value) in fields) {
                val reference = expected.optString(key)
                if (reference != value) {
                    mismatches.put(
                        JSONObject()
                            .put("text", text)
                            .put("attribute", key)
                            .put("python", reference)
                            .put("android", value),
                    )
                }
            }
        }
        return mismatches
    }

    private fun runBenchmark(
        backend: String,
        modelAsset: String,
        runs: Int,
        profileOperators: Boolean,
    ): JSONObject {
        require(backend == "cpu" || backend == "nnapi") {
            "backend must be cpu or nnapi"
        }
        require(modelAsset in MODEL_ASSETS) { "unknown model asset" }
        if (modelAsset == SELECTIVE_PIPELINE) {
            require(backend == "cpu") { "selective pipeline supports cpu only" }
            require(!profileOperators) { "operator profiling is not yet supported for pipeline" }
            return runSelectivePipeline(runs)
        }
        val modelBytes = assets.open(modelAsset).use { it.readBytes() }
        val environment = OrtEnvironment.getEnvironment()
        val options = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(CPU_THREADS)
            setInterOpNumThreads(1)
            if (backend == "nnapi") {
                addNnapi()
            }
            if (profileOperators) {
                enableProfiling(File(filesDir, "ort-profile").absolutePath)
            }
        }

        val memoryBeforeKb = Debug.getPss()
        val sessionStarted = SystemClock.elapsedRealtimeNanos()
        val session = environment.createSession(modelBytes, options)
        val sessionMilliseconds = elapsedMilliseconds(sessionStarted)

        val tokenBuffer = ByteBuffer
            .allocateDirect(MAXIMUM_TOKENS * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer()
        repeat(MAXIMUM_TOKENS) { tokenBuffer.put(0L) }
        tokenBuffer.rewind()

        val latentElements = LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE
        val latentBuffer = ByteBuffer
            .allocateDirect(latentElements * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        repeat(latentElements) { latentBuffer.put(0.0f) }
        latentBuffer.rewind()

        val tokenTensor = OnnxTensor.createTensor(
            environment,
            tokenBuffer,
            longArrayOf(1, MAXIMUM_TOKENS.toLong()),
        )
        val latentTensor = OnnxTensor.createTensor(
            environment,
            latentBuffer,
            longArrayOf(1, LATENT_CHANNELS.toLong(), LATENT_SIZE.toLong(), LATENT_SIZE.toLong()),
        )
        val conditionElements = MODEL_WIDTH
        val conditionBuffer = ByteBuffer
            .allocateDirect(conditionElements * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        repeat(conditionElements) { conditionBuffer.put(0.0f) }
        conditionBuffer.rewind()
        val conditionTensor = OnnxTensor.createTensor(
            environment,
            conditionBuffer,
            longArrayOf(1, MODEL_WIDTH.toLong()),
        )
        val studentElements = 3 * STUDENT_IMAGE_SIZE * STUDENT_IMAGE_SIZE
        val studentSampleBuffer = ByteBuffer
            .allocateDirect(studentElements * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        repeat(studentElements) { studentSampleBuffer.put(0.0f) }
        studentSampleBuffer.rewind()
        val studentSampleTensor = OnnxTensor.createTensor(
            environment,
            studentSampleBuffer,
            longArrayOf(1, 3, STUDENT_IMAGE_SIZE.toLong(), STUDENT_IMAGE_SIZE.toLong()),
        )
        val studentRatioBuffer = ByteBuffer
            .allocateDirect(java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        studentRatioBuffer.put(0.5f)
        studentRatioBuffer.rewind()
        val studentRatioTensor = OnnxTensor.createTensor(
            environment,
            studentRatioBuffer,
            longArrayOf(1),
        )
        val studentAttributesBuffer = ByteBuffer
            .allocateDirect(STUDENT_ATTRIBUTES * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer()
        repeat(STUDENT_ATTRIBUTES) { studentAttributesBuffer.put(0L) }
        studentAttributesBuffer.rewind()
        val studentAttributesTensor = OnnxTensor.createTensor(
            environment,
            studentAttributesBuffer,
            longArrayOf(1, STUDENT_ATTRIBUTES.toLong()),
        )
        val availableInputs = mapOf(
            "token_ids" to tokenTensor,
            "latent" to latentTensor,
            "condition" to conditionTensor,
            "sample" to studentSampleTensor,
            "ratio" to studentRatioTensor,
            "attributes" to studentAttributesTensor,
        )
        val inputNames = session.inputNames
        val inputs = inputNames.associateWith { name ->
            requireNotNull(availableInputs[name]) { "unsupported model input: $name" }
        }

        session.run(inputs).use { warmup -> checksum(warmup) }
        val durations = JSONArray()
        var checksum = 0.0
        var maximumPssKb = Debug.getPss()
        repeat(runs) {
            val started = SystemClock.elapsedRealtimeNanos()
            val result = session.run(inputs)
            val inferenceMilliseconds = elapsedMilliseconds(started)
            result.use { checksum = checksum(it) }
            durations.put(inferenceMilliseconds)
            maximumPssKb = maxOf(maximumPssKb, Debug.getPss())
        }

        val memoryAfterKb = Debug.getPss()
        val profileFile = if (profileOperators) File(session.endProfiling()).name else null
        tokenTensor.close()
        latentTensor.close()
        conditionTensor.close()
        studentSampleTensor.close()
        studentRatioTensor.close()
        studentAttributesTensor.close()
        session.close()
        options.close()

        return JSONObject()
            .put("schema_version", 1)
            .put("status", "ok")
            .put("model", modelAsset)
            .put("model_bytes", modelBytes.size)
            .put("input_names", JSONArray(inputNames))
            .put("backend_requested", backend)
            .put("runs", runs)
            .put("session_creation_ms", sessionMilliseconds)
            .put("durations_ms", durations)
            .put("median_ms", median(durations))
            .put("pss_before_kb", memoryBeforeKb)
            .put("pss_after_kb", memoryAfterKb)
            .put("pss_maximum_sampled_kb", maximumPssKb)
            .put("checksum", checksum)
            .put("ort_version", environment.version)
            .put("cpu_threads", CPU_THREADS)
            .put("profile_file", profileFile ?: JSONObject.NULL)
    }

    private fun runSelectivePipeline(runs: Int): JSONObject {
        val environment = OrtEnvironment.getEnvironment()
        val encoderBytes = assets.open(SELECTIVE_ENCODER).use { it.readBytes() }
        val denoiserBytes = assets.open(SELECTIVE_DENOISER).use { it.readBytes() }
        val decoderBytes = assets.open(SELECTIVE_DECODER).use { it.readBytes() }
        val memoryBeforeKb = Debug.getPss()

        fun sessionOptions() = OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(CPU_THREADS)
            setInterOpNumThreads(1)
        }

        val encoderOptions = sessionOptions()
        val encoderStarted = SystemClock.elapsedRealtimeNanos()
        val encoderSession = environment.createSession(encoderBytes, encoderOptions)
        val encoderSessionMs = elapsedMilliseconds(encoderStarted)

        val denoiserOptions = sessionOptions()
        val denoiserStarted = SystemClock.elapsedRealtimeNanos()
        val denoiserSession = environment.createSession(denoiserBytes, denoiserOptions)
        val denoiserSessionMs = elapsedMilliseconds(denoiserStarted)

        val decoderOptions = sessionOptions()
        val decoderStarted = SystemClock.elapsedRealtimeNanos()
        val decoderSession = environment.createSession(decoderBytes, decoderOptions)
        val decoderSessionMs = elapsedMilliseconds(decoderStarted)

        val tokenBuffer = ByteBuffer
            .allocateDirect(MAXIMUM_TOKENS * java.lang.Long.BYTES)
            .order(ByteOrder.nativeOrder())
            .asLongBuffer()
        repeat(MAXIMUM_TOKENS) { tokenBuffer.put(0L) }
        tokenBuffer.rewind()
        val tokenTensor = OnnxTensor.createTensor(
            environment,
            tokenBuffer,
            longArrayOf(1, MAXIMUM_TOKENS.toLong()),
        )

        val latentElements = LATENT_CHANNELS * LATENT_SIZE * LATENT_SIZE
        val latentBuffer = ByteBuffer
            .allocateDirect(latentElements * java.lang.Float.BYTES)
            .order(ByteOrder.nativeOrder())
            .asFloatBuffer()
        repeat(latentElements) { latentBuffer.put(0.0f) }
        latentBuffer.rewind()
        val latentTensor = OnnxTensor.createTensor(
            environment,
            latentBuffer,
            longArrayOf(1, LATENT_CHANNELS.toLong(), LATENT_SIZE.toLong(), LATENT_SIZE.toLong()),
        )

        val conditionStarted = SystemClock.elapsedRealtimeNanos()
        val encoderResult = encoderSession.run(mapOf("token_ids" to tokenTensor))
        val encoderOnceMs = elapsedMilliseconds(conditionStarted)
        val conditionTensor = encoderResult[0] as OnnxTensor

        fun executePipeline(measure: Boolean): Pair<Double, Double> {
            val started = SystemClock.elapsedRealtimeNanos()
            val denoiserResult = denoiserSession.run(
                mapOf("condition" to conditionTensor, "latent" to latentTensor),
            )
            val predictedLatent = denoiserResult[0] as OnnxTensor
            val decoderResult = decoderSession.run(mapOf("latent" to predictedLatent))
            val duration = if (measure) elapsedMilliseconds(started) else 0.0
            val checksum = checksum(decoderResult)
            decoderResult.close()
            denoiserResult.close()
            return Pair(duration, checksum)
        }

        executePipeline(measure = false)
        val durations = JSONArray()
        var checksum = 0.0
        var maximumPssKb = Debug.getPss()
        repeat(runs) {
            val result = executePipeline(measure = true)
            durations.put(result.first)
            checksum = result.second
            maximumPssKb = maxOf(maximumPssKb, Debug.getPss())
        }
        val memoryAfterKb = Debug.getPss()

        encoderResult.close()
        tokenTensor.close()
        latentTensor.close()
        encoderSession.close()
        denoiserSession.close()
        decoderSession.close()
        encoderOptions.close()
        denoiserOptions.close()
        decoderOptions.close()

        return JSONObject()
            .put("schema_version", 1)
            .put("status", "ok")
            .put("model", SELECTIVE_PIPELINE)
            .put("model_bytes", encoderBytes.size + denoiserBytes.size + decoderBytes.size)
            .put(
                "component_model_bytes",
                JSONObject()
                    .put("encoder", encoderBytes.size)
                    .put("denoiser", denoiserBytes.size)
                    .put("decoder", decoderBytes.size),
            )
            .put("backend_requested", "cpu")
            .put("runs", runs)
            .put(
                "component_session_creation_ms",
                JSONObject()
                    .put("encoder", encoderSessionMs)
                    .put("denoiser", denoiserSessionMs)
                    .put("decoder", decoderSessionMs),
            )
            .put("session_creation_ms", encoderSessionMs + denoiserSessionMs + decoderSessionMs)
            .put("encoder_once_ms", encoderOnceMs)
            .put("encoder_cached", true)
            .put("durations_ms", durations)
            .put("median_ms", median(durations))
            .put("pss_before_kb", memoryBeforeKb)
            .put("pss_after_kb", memoryAfterKb)
            .put("pss_maximum_sampled_kb", maximumPssKb)
            .put("checksum", checksum)
            .put("ort_version", environment.version)
            .put("cpu_threads", CPU_THREADS)
            .put("profile_file", JSONObject.NULL)
    }

    private fun checksum(result: OrtSession.Result): Double {
        val tensor = result[0] as OnnxTensor
        val buffer = tensor.floatBuffer
        var total = 0.0
        while (buffer.hasRemaining()) {
            total += buffer.get().toDouble()
        }
        return total
    }

    private fun median(values: JSONArray): Double {
        val sorted = (0 until values.length()).map(values::getDouble).sorted()
        val middle = sorted.size / 2
        return if (sorted.size % 2 == 0) {
            (sorted[middle - 1] + sorted[middle]) / 2.0
        } else {
            sorted[middle]
        }
    }

    private fun elapsedMilliseconds(started: Long): Double =
        (SystemClock.elapsedRealtimeNanos() - started) / 1_000_000.0

    companion object {
        private const val TAG = "AvatarFaceBenchmark"
        private const val DEFAULT_MODEL_ASSET = "avatarface-feasibility-micro.onnx"
        private const val SELECTIVE_PIPELINE = "avatarface-feasibility-bridge-selective.onnx"
        private const val SELECTIVE_ENCODER = "avatarface-feasibility-bridge-encoder.onnx"
        private const val SELECTIVE_DENOISER =
            "avatarface-feasibility-bridge-denoiser-int8-preprocessed.onnx"
        private const val SELECTIVE_DECODER =
            "avatarface-feasibility-bridge-fast-decoder.onnx"
        private val MODEL_ASSETS = setOf(
            DEFAULT_MODEL_ASSET,
            SELECTIVE_PIPELINE,
            "avatarface-feasibility-micro-int8.onnx",
            "avatarface-feasibility-bridge.onnx",
            "avatarface-feasibility-bridge-int8.onnx",
            "avatarface-feasibility-bridge-int8-preprocessed.onnx",
            "avatarface-feasibility-bridge-encoder.onnx",
            "avatarface-feasibility-bridge-denoiser.onnx",
            "avatarface-feasibility-bridge-decoder.onnx",
            "avatarface-feasibility-bridge-denoiser-int8-preprocessed.onnx",
            "avatarface-feasibility-bridge-decoder-int8-preprocessed.onnx",
            "avatarface-feasibility-bridge-slim-decoder.onnx",
            "avatarface-feasibility-bridge-slim-decoder-int8-preprocessed.onnx",
            "avatarface-feasibility-bridge-fast-decoder.onnx",
            "avatarface-feasibility-bridge-fast-decoder-int8-preprocessed.onnx",
            "avatarface-feasibility-bridge-fast.onnx",
            STUDENT_MODEL,
            "avatarface-student.onnx",
            "avatarface-student-lite-int8.onnx",
            "avatarface-student-lite-final-int8.onnx",
            "avatarface-student-lite-fp32.onnx",
            "avatarface-student-lite-a16.onnx",
            "avatarface-student-lite24-int8.onnx",
        )
        // Estudiante del ADR 0010: un paso del U-Net; la app ejecuta la
        // cadena DDIM completa multiplicando por STUDENT_DDIM_STEPS.
        private const val STUDENT_MODEL = "avatarface-student-int8.onnx"
        private const val STUDENT_IMAGE_SIZE = 256
        private const val STUDENT_ATTRIBUTES = 9
        private const val GALLERY_SPECS = "gallery-specs.json"
        private const val PARSER_CASES = "parser-cases.json"
        private const val AVATAR_SIZE = 256
        private const val RESULT_FILE = "benchmark-result.json"
        private const val DEFAULT_RUNS = 5
        private const val MAX_RUNS = 50
        private const val CPU_THREADS = 6
        private const val MAXIMUM_TOKENS = 16
        private const val LATENT_CHANNELS = 4
        private const val LATENT_SIZE = 8
        private const val MODEL_WIDTH = 320
    }
}
