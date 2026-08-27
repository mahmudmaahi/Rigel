import next from "eslint-config-next";

const eslintConfig = [
  {
    ignores: [".next/**", "out/**", "node_modules/**"],
  },
  ...next,
];

export default eslintConfig;
