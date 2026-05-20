# Android Build Environment Dockerfile
# 최신 버전 (2025년 기준)

FROM ubuntu:24.04

LABEL maintainer="MathAI Team"
LABEL description="Android build environment for Math AI app"

# 환경 변수 설정
ENV DEBIAN_FRONTEND=noninteractive
ENV ANDROID_HOME=/opt/android-sdk
ENV ANDROID_SDK_ROOT=/opt/android-sdk
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH="${PATH}:${ANDROID_HOME}/cmdline-tools/latest/bin:${ANDROID_HOME}/platform-tools:${ANDROID_HOME}/build-tools/34.0.0"

# 기본 패키지 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    wget \
    unzip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Android SDK 설치
RUN mkdir -p ${ANDROID_HOME}/cmdline-tools && \
    cd ${ANDROID_HOME}/cmdline-tools && \
    wget -q https://dl.google.com/android/repository/commandlinetools-linux-11076708_latest.zip -O cmdline-tools.zip && \
    unzip -q cmdline-tools.zip && \
    mv cmdline-tools latest && \
    rm cmdline-tools.zip

# Android SDK 라이선스 동의 및 컴포넌트 설치
RUN yes | sdkmanager --licenses && \
    sdkmanager \
    "platform-tools" \
    "platforms;android-36" \
    "build-tools;36.0.0" \
    "build-tools;34.0.0"

# Gradle 캐시 디렉토리 설정
ENV GRADLE_USER_HOME=/root/.gradle

# 작업 디렉토리 설정
WORKDIR /app

# Gradle Wrapper 복사 및 의존성 사전 다운로드 (캐싱 최적화)
COPY gradlew .
COPY gradle gradle
COPY build.gradle.kts .
COPY settings.gradle.kts .
COPY gradle.properties .
COPY gradle/libs.versions.toml gradle/

# gradlew 실행 권한 부여
RUN chmod +x gradlew

# 전체 소스 복사
COPY . .

# 빌드 명령어 (기본값: debug APK 빌드)
CMD ["./gradlew", "assembleDebug", "--no-daemon"]