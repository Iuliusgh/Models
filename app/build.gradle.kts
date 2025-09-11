plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.jetbrains.kotlin.android)
    alias(libs.plugins.compose.compiler)
}

android {
    namespace = "com.example.models"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.example.models"
        minSdk = 33
        targetSdk = 34
        versionCode = 1
        versionName = "1.0"
        vectorDrawables {
            useSupportLibrary = true
        }
        ndk { abiFilters.add("arm64-v8a") }
        packaging { jniLibs { useLegacyPackaging = true } }
    }
    dataBinding{
        enable = true
    }
    buildTypes {
        debug{
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            ndk { abiFilters.add("arm64-v8a") }
        }
        release {
            isMinifyEnabled = false
            proguardFiles(getDefaultProguardFile("proguard-android-optimize.txt"), "proguard-rules.pro")
            ndk { abiFilters.add("arm64-v8a") }
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    buildFeatures {
        compose = true
        mlModelBinding = true
        viewBinding = true
    }
    composeOptions {
        kotlinCompilerExtensionVersion = "1.5.1"
    }
    packaging {
        resources {
            excludes += "/META-INF/{AL2.0,LGPL2.1}"
        }
        jniLibs.useLegacyPackaging = true
    }
    //ndkVersion = "28.0.12433566"
    /*sourceSets{
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }*/
    dynamicFeatures.add(":litert_npu_runtime_libraries:qualcomm_runtime_v73")
}

dependencies {
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.ui)
    implementation(libs.androidx.ui.graphics)
    implementation(libs.androidx.ui.tooling.preview)
    implementation(libs.androidx.material3)
    implementation(libs.androidx.fragment.ktx)
    implementation(libs.androidx.constraintlayout)
    implementation(libs.androidx.constraintlayout.compose)
    implementation(libs.androidx.appcompat)
    implementation(project(":opencv"))
    implementation(project(":litert_npu_runtime_libraries:runtime_strings"))
    implementation(libs.androidx.coordinatorlayout)
    implementation(libs.kotlinx.serialization.json)
    debugImplementation(libs.androidx.ui.tooling)
    debugImplementation(libs.androidx.ui.test.manifest)
    implementation(libs.qnn.runtime)
    implementation(libs.qnn.litert.delegate)
    implementation(libs.litert)
    implementation(libs.litert.gpu)
    implementation(libs.litert.api)
    implementation(libs.litert.metadata)
    implementation(libs.litert.support)
    //implementation(files("../libs/qtld-release.aar"))
    //implementation(files("../libs/platform-validator.aar"))
    //implementation(files("../libs/snpe-release.aar"))

}