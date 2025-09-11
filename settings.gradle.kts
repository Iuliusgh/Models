pluginManagement {
    repositories {
        google()
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "Models"
include(":app")
include(":opencv")
include(":litert_npu_runtime_libraries:runtime_strings")
include(":litert_npu_runtime_libraries:qualcomm_runtime_v73")

