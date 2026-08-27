import { clsx } from "clsx";

type CardProps = React.HTMLAttributes<HTMLDivElement>;

export function Card({ className, ...props }: CardProps) {
  return (
    <div
      className={clsx("border border-border bg-card text-card-foreground shadow-sm", className)}
      {...props}
    />
  );
}
