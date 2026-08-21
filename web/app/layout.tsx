import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Tracking AI Collector",
  description: "Standee audience tracking collector",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="vi">
      <body>{children}</body>
    </html>
  );
}
