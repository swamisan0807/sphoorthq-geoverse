# Real, buildable image for the web UI - validated in CI. Static build served
# by nginx. VITE_API_BASE must point at wherever src/api's container/service
# actually runs - baked in at build time via --build-arg.
FROM node:20-slim AS build
WORKDIR /app
COPY package.json package-lock.json ./
RUN npm ci
COPY . .
ARG VITE_API_BASE=http://localhost:8000
ENV VITE_API_BASE=${VITE_API_BASE}
RUN npm run build

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
