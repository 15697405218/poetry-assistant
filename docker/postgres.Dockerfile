# 用本地 pgvector 源码编进 postgres:16-alpine，避免再拉镜像/git clone（国内常 502）。
# 数据目录仍挂原来的 volume，不要 down -v。
# with_llvm=no：官方镜像开了 LLVM，容器里没有 clang 会编 .bc 失败；.so 不需要 bitcode。
FROM postgres:16-alpine

COPY docker/pgvector-src/pgvector-0.8.0 /tmp/pgvector
RUN apk add --no-cache --virtual .build-deps build-base \
    && make -C /tmp/pgvector OPTFLAGS="" with_llvm=no \
    && make -C /tmp/pgvector install with_llvm=no \
    && rm -rf /tmp/pgvector \
    && apk del .build-deps
