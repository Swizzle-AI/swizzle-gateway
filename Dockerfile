# ==============================================================================
# Swizzle Gateway — Multi-Stage Production Dockerfile (<15MB static image)
# ==============================================================================

FROM golang:1.22-alpine AS builder
WORKDIR /build

COPY go.mod ./
COPY *.go ./
COPY static ./static

RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o swizzle-gateway .

FROM alpine:3.20
RUN apk --no-cache add ca-certificates tzdata
WORKDIR /app

COPY --from=builder /build/swizzle-gateway /app/swizzle-gateway
COPY --from=builder /build/static /app/static

EXPOSE 8080
ENV PORT=8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=2s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://127.0.0.1:8080/v1/telemetry || exit 1

ENTRYPOINT ["/app/swizzle-gateway"]
