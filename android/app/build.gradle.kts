plugins {
    id("com.android.application")
}

android {
    namespace = "com.avatarface.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.avatarface.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0"
        ndk {
            abiFilters += "arm64-v8a"
        }
    }

    buildTypes {
        debug {
            isMinifyEnabled = false
        }
        release {
            isMinifyEnabled = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro",
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        getByName("main").assets.directories.add(
            "../../artifacts/feasibility",
        )
    }

    androidResources {
        noCompress += "onnx"
    }
}

dependencies {
    implementation("com.microsoft.onnxruntime:onnxruntime-android:1.23.2")
}
