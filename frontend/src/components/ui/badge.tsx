import { clsx } from "clsx";

type BadgeProps = {
  children: React.ReactNode;
  variant?: "solid" | "outline";
};

export function Badge({ children, variant = "solid" }: BadgeProps) {
  return (
    <span
      className={clsx(
        "inline-flex h-8 items-center border px-3 text-xs font-medium uppercase tracking-[0.14em]",
        variant === "solid" && "border-primary bg-primary text-primary-foreground",
        variant === "outline" && "border-border bg-transparent text-foreground",
      )}
    >
      {children}
    </span>
  );
}
