// Server-side proxy to the FastAPI backend. The browser calls /api/backend/*
// (same-origin, no CORS) and this handler forwards to the backend, injecting the
// x-api-key from a SERVER-ONLY env var so the key never reaches the client.

import { NextRequest, NextResponse } from "next/server";

export const dynamic = "force-dynamic";

const BACKEND_URL = (process.env.BACKEND_URL || "http://localhost:3001").replace(/\/$/, "");
const API_KEY = process.env.API_KEY || "changeme-set-a-real-key";

async function forward(req: NextRequest, path: string[]) {
  const search = req.nextUrl.search || "";
  const target = `${BACKEND_URL}/api/${path.join("/")}${search}`;

  const headers: Record<string, string> = { "x-api-key": API_KEY };
  const contentType = req.headers.get("content-type");
  // Let fetch set the multipart boundary itself; pass through other types.
  if (contentType && !contentType.includes("multipart/form-data")) {
    headers["content-type"] = contentType;
  }

  const method = req.method.toUpperCase();
  let body: BodyInit | undefined;
  if (method !== "GET" && method !== "HEAD") {
    if (contentType && contentType.includes("multipart/form-data")) {
      body = await req.formData();
    } else {
      body = await req.text();
    }
  }

  let resp: Response;
  try {
    resp = await fetch(target, { method, headers, body, cache: "no-store" });
  } catch (e) {
    return NextResponse.json(
      { error: "Backend unreachable", detail: String(e) },
      { status: 502 },
    );
  }

  const respType = resp.headers.get("content-type") || "application/octet-stream";
  if (respType.startsWith("image/")) {
    const buf = await resp.arrayBuffer();
    return new NextResponse(buf, {
      status: resp.status,
      headers: { "content-type": respType },
    });
  }

  const text = await resp.text();
  return new NextResponse(text, {
    status: resp.status,
    headers: { "content-type": respType },
  });
}

type Ctx = { params: { path: string[] } };

export async function GET(req: NextRequest, { params }: Ctx) {
  return forward(req, params.path);
}
export async function POST(req: NextRequest, { params }: Ctx) {
  return forward(req, params.path);
}
export async function PUT(req: NextRequest, { params }: Ctx) {
  return forward(req, params.path);
}
export async function PATCH(req: NextRequest, { params }: Ctx) {
  return forward(req, params.path);
}
export async function DELETE(req: NextRequest, { params }: Ctx) {
  return forward(req, params.path);
}
