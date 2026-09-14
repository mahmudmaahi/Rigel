import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Rigel | Audio Signal Processing Platform by Team Orion",
  description:
    "From raw signals to pure clarity. Professional audio analysis with discrete-time DSP, real-time waveform rendering, and peak envelope visualization. Open source audio platform by Team Orion.",
  icons: {
    icon: "/logo.png",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`antialiased`}
        style={{ fontFamily: "'Century Gothic', sans-serif" }}
      >
        {children}
      </body>
    </html>
  );
}
