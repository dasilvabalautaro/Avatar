package com.avatarface.app

import ai.onnxruntime.OnnxTensor
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
        Thread {
            val payload = try {
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
        val availableInputs = mapOf(
            "token_ids" to tokenTensor,
            "latent" to latentTensor,
            "condition" to conditionTensor,
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
        )
        private const val RESULT_FILE = "benchmark-result.json"
        private const val DEFAULT_RUNS = 5
        private const val MAX_RUNS = 50
        private const val CPU_THREADS = 4
        private const val MAXIMUM_TOKENS = 16
        private const val LATENT_CHANNELS = 4
        private const val LATENT_SIZE = 8
        private const val MODEL_WIDTH = 320
    }
}
