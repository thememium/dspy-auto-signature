# Changelog

## v0.1.1 (2026-06-06)

### 🚀 Enhancements

- **dspy_auto_signature**: add generate() API and validate hint names ([191369c](https://github.com/thememium/dspy-auto-signature/commit/191369c8b79cc3d739f76c3094b1f37c56daf50c))
- **parser**: support dict inputs and enrich parse results with metadata ([0621e87](https://github.com/thememium/dspy-auto-signature/commit/0621e871a37b9acf342d0aa9acdb47c159d5a314))
- add source metadata fields to ParsedPrompt ([4b60030](https://github.com/thememium/dspy-auto-signature/commit/4b600306ab12c08b517060422d9446f6b0a6d276))
- **type_resolver**: add support for Python-style generics and union types ([0cb3bb8](https://github.com/thememium/dspy-auto-signature/commit/0cb3bb84c54eb918d7c2302985fb18e66934e43b))
- **type-resolver**: add literal, dict, union support and Pydantic models ([1f9c4b4](https://github.com/thememium/dspy-auto-signature/commit/1f9c4b45cc642abab979be4acfd8311c042f447d))
- **example**: add dataset-driven example script for dspy-auto-signature ([a9e9145](https://github.com/thememium/dspy-auto-signature/commit/a9e91450cf985a14651b24941e8c41b8d8f05907))
- **dspy_auto_signature**: add dataset‑grounded signature generator and extend configure ([fb5aa30](https://github.com/thememium/dspy-auto-signature/commit/fb5aa30fe3857ed768cb71c889b4a79954c6f4bf))
- **parser**: add DataFrameParser to AutoParser ([bc62203](https://github.com/thememium/dspy-auto-signature/commit/bc62203baf3e9044b474739bfffce295fe1aba66))
- add RLM generator for DSPy signatures ([caab9de](https://github.com/thememium/dspy-auto-signature/commit/caab9de6b747e1686071be1f108f3cb5c95cecc0))
- **dspy_auto_signature**: add RLM-based slow‑path signature generator ([ad7eca4](https://github.com/thememium/dspy-auto-signature/commit/ad7eca4a8032397ca1ffbfc64c8e8879d37cedba))
- **data**: add profiler and to_records modules for column profiling and record conversion ([b7a2269](https://github.com/thememium/dspy-auto-signature/commit/b7a226991d31012467b3399584c755970c430561))
- **core/config.py**: add dataset_lm and sub_lm configuration options ([0d87caa](https://github.com/thememium/dspy-auto-signature/commit/0d87caa2e2b032ad40ef63bbbe4c30770c5d6036))
- **parser**: add DataFrameParser for tabular data ([3fc7b7e](https://github.com/thememium/dspy-auto-signature/commit/3fc7b7e1de75a5a2e12c8f5e434465d5d2680a43))
- **example.py**: persist signature to file and use ChainOfThought with signature ([389c0d3](https://github.com/thememium/dspy-auto-signature/commit/389c0d30d3d4ed60d5e5aad1ae5e0070f6abc2a3))
- **signature_builder**: add source generation capability ([df9d1e9](https://github.com/thememium/dspy-auto-signature/commit/df9d1e940831036fd059c9bfd671e8fcb53da878))
- **dspy_auto_signature**: expose public API for signature generation ([36d14a1](https://github.com/thememium/dspy-auto-signature/commit/36d14a198f067357b7877c62604c7643e92f0cca))
- **parser**: add AutoParser to orchestrate multiple prompt parsers ([86abdc9](https://github.com/thememium/dspy-auto-signature/commit/86abdc9557049502d4472231ce7d9716536b3ac5))
- **parser**: add VercelParser to parse Vercel AI SDK messages ([f325ff1](https://github.com/thememium/dspy-auto-signature/commit/f325ff1a7ef6fab75da5c583655a8c1319268fd7))
- **parser**: add StringParser for raw string prompts ([6213ae1](https://github.com/thememium/dspy-auto-signature/commit/6213ae158578d149578483ba5abbb8f7e1cfac4f))
- **parser**: add abstract PromptParser base class ([dfa7ac7](https://github.com/thememium/dspy-auto-signature/commit/dfa7ac797255d088367b26342324eceaa8eaaf1c))
- **dspy_auto_signature**: add type resolver utility for natural language type mapping ([ccc9b41](https://github.com/thememium/dspy-auto-signature/commit/ccc9b41416b735a2011971aa1d8f4b6ddc6738e7))
- **dspy_auto_signature**: add signature specification types ([4e709f7](https://github.com/thememium/dspy-auto-signature/commit/4e709f767c76a6357c24d16248b943eeaec2f2a7))
- **prompts**: add system prompts for signature generation ([86c7900](https://github.com/thememium/dspy-auto-signature/commit/86c7900693edd2fed667666fd9793684b5578543))
- **signature_generator**: add SignatureGenerator to generate SignatureSpec ([4abd6c4](https://github.com/thememium/dspy-auto-signature/commit/4abd6c4412d3c2014494dbbf8d076072f22413af))
- **signature**: add SignatureBuilder to generate DSPy Signature classes ([56546ef](https://github.com/thememium/dspy-auto-signature/commit/56546ef903418bbd95a42210ab37cb11949d08cf))
- **core**: add global configuration for language model ([d6b7a41](https://github.com/thememium/dspy-auto-signature/commit/d6b7a41e21353bd0f4b04d980baa0b07f64d108d))

### 💅 Refactors

- **example**: simplify support ticket prompt ([cd0c414](https://github.com/thememium/dspy-auto-signature/commit/cd0c414a282a00136a35f31c7db56de6547a4562))
- unify signature generation API and update docs ([5e380bb](https://github.com/thememium/dspy-auto-signature/commit/5e380bbeb942511b8b6b3172a2cf9c101859c73f))
- **example.py**: rename das.from_prompt to das.generate ([d6e8e1c](https://github.com/thememium/dspy-auto-signature/commit/d6e8e1c68a90b28ea826b6205e25d1ef698691a3))
- **example_dataset.py**: switch example to das.generate ([a1f5616](https://github.com/thememium/dspy-auto-signature/commit/a1f5616b33d5f3127e8dd6958d10f08587dc1caf))
- **public-api**: add generate, enforce from_dataset type safety ([1610823](https://github.com/thememium/dspy-auto-signature/commit/16108232822ef9fa652dc2a6cba8be3bcf30051f))
- **to_records.py**: add Mapping support and improve list handling ([6df1f8f](https://github.com/thememium/dspy-auto-signature/commit/6df1f8fb39331918c6f609a06d102a555c876b94))
- **generator**: expose RLMSignatureGenerator and use prompt data_profile ([def2e81](https://github.com/thememium/dspy-auto-signature/commit/def2e81117d0c06314cf1d5c471a016799d96110))
- **example**: improve RLM meta-model configuration and output paths ([e746591](https://github.com/thememium/dspy-auto-signature/commit/e74659191e40e016d5118c2ab38db8e0c4c6af2e))
- **dspy_auto_signature**: switch to RLMSignatureGenerator ([20d9e0d](https://github.com/thememium/dspy-auto-signature/commit/20d9e0de129fbea6a1b798ba5983028c7efc5dbe))
- **rlm_signatures**: unify RLM signature generator ([b61b7c4](https://github.com/thememium/dspy-auto-signature/commit/b61b7c4371986f3c98a1f0dee945f0903d525f27))
- **rlm_signature_generator.py**: unify RLM‑based signature generator and add robust parsing ([4d16222](https://github.com/thememium/dspy-auto-signature/commit/4d1622247278441e1b3ba7820683883c5f3c1dab))
- **dspy_auto_signature**: improve type conversion and import handling for generated signatures ([0ba2d55](https://github.com/thememium/dspy-auto-signature/commit/0ba2d55ff4f646e4db20426a9c271d55c215e475))
- **dspy_auto_signature**: clean up inline comments ([134798e](https://github.com/thememium/dspy-auto-signature/commit/134798e2d69881e96a66d3dae4b954ec34c1507f))
- **signature_generator**: simplify field extraction and parallelize analysis ([e62d6fe](https://github.com/thememium/dspy-auto-signature/commit/e62d6feb9644741aff9e3fd8176fe40440b1e77c))
- **dspy_auto_signature**: expose GeneratedSignature ([016196f](https://github.com/thememium/dspy-auto-signature/commit/016196fadd6f2022d41dde15acd47e9c4b2ff8ae))
- **dspy_auto_signature**: add type casting and update generator call ([349353d](https://github.com/thememium/dspy-auto-signature/commit/349353db97d41d214c61eb6520319bde0f33f5aa))
- **signature_generator**: provide default descriptions for missing fields ([253297a](https://github.com/thememium/dspy-auto-signature/commit/253297a90e3ed007bc772f113b6f0bb327a322c9))
- **signature_builder**: add field‑description validation and metadata update ([78064d6](https://github.com/thememium/dspy-auto-signature/commit/78064d6df95ddc8217262cb266f85bddfddfab7b))
- **signature_generator**: isolate LM context using Config.get_lm() ([4ce465f](https://github.com/thememium/dspy-auto-signature/commit/4ce465fadeeb7147ab5bd819ad2824818ac8da88))
- delete unused prompt files ([192d652](https://github.com/thememium/dspy-auto-signature/commit/192d6522dbdfd7e9672f9a8a57b0e15e8f61ead5))

### 📖 Documentation

- **signature_generator.py**: improve class docstrings for clarity ([07c0ad1](https://github.com/thememium/dspy-auto-signature/commit/07c0ad1e4d1831293449fce06dcd74227c7653c7))
- **core/config.py**: clarify LM docstrings and get_dataset_lm description ([59586ef](https://github.com/thememium/dspy-auto-signature/commit/59586efcad2731bbccce96d61fa8ad23e76b4526))
- **readme**: add dataset input slow path section and update configure docs ([13f4abf](https://github.com/thememium/dspy-auto-signature/commit/13f4abf634ae7eecb26037c8dd8e7488db7be88f))
- update README examples to use OpenRouter GPT-OSS-120b ([a2d218b](https://github.com/thememium/dspy-auto-signature/commit/a2d218bc1614aa06dff1cbf1ed9ef74352ad33dc))
- **example**: add e2e example for dspy-auto-signature ([eff68db](https://github.com/thememium/dspy-auto-signature/commit/eff68db9a9021078aef593ebb18acd50455678e6))
- overhaul README to include header, TOC, sections, and expanded content ([8842166](https://github.com/thememium/dspy-auto-signature/commit/8842166cf578bcba53efa0362b23633ebfeec164))
- **README**: clarify meta‑model vs runtime model usage ([1c5c760](https://github.com/thememium/dspy-auto-signature/commit/1c5c7608b04a02afb6d7e84caaff7e168872dcb9))
- add complete README with usage, API, and architecture details ([0e32b64](https://github.com/thememium/dspy-auto-signature/commit/0e32b642753bcaa213d9674d1760d7986724e881))
- **plan.md**: remove generator/prompts dir and update module list ([3c35624](https://github.com/thememium/dspy-auto-signature/commit/3c356240c26f46911afa2b446a640c05139e9b5c))
- **plan.md**: add design plan for DSPy auto‑signature generator ([d7dbebf](https://github.com/thememium/dspy-auto-signature/commit/d7dbebfd081a85b1b54fb46002a6977b1e9e13ac))

### 📦 Build

- **pyproject.toml**: add pandas as a runtime dependency ([5eef2bf](https://github.com/thememium/dspy-auto-signature/commit/5eef2bf2c1f3add9039e381af000795078b9e53e))
- **pyproject**: add dspy and pydantic dependencies ([9a3462b](https://github.com/thememium/dspy-auto-signature/commit/9a3462b47654dead74661f9b72d52d1ae5355068))

### 🏡 Chore

- **deps**: bump usechange to 0.1.35 ([5b5d5d1](https://github.com/thememium/dspy-auto-signature/commit/5b5d5d133529ab82168b35cc6118671546a32d76))
- **pyproject.toml**: remove dspy-auto-signature script entry ([1d66c55](https://github.com/thememium/dspy-auto-signature/commit/1d66c55eb3a0cdc76534b279db0f5457de430885))
- remove deprecated signature generator module ([41b444f](https://github.com/thememium/dspy-auto-signature/commit/41b444f1fb447bc6f1702aae84604c2d74397a18))
- update .gitignore to ignore output dir and drop *_signature.py rule ([d39831d](https://github.com/thememium/dspy-auto-signature/commit/d39831dc00d12e29b696b46ef52a99d95ee636a6))
- **gitignore**: ignore *_signature.py files ([21dd6d3](https://github.com/thememium/dspy-auto-signature/commit/21dd6d36086bafdab37f7bd794c3c519b3f016a3))
- **pyproject**: reorganize dependencies and add optional dataset support ([75116bc](https://github.com/thememium/dspy-auto-signature/commit/75116bc6bdab3b16c8d4f024641e2895f6e4cfbb))
- **.gitignore**: add .omo to ignore generated files ([4a05f75](https://github.com/thememium/dspy-auto-signature/commit/4a05f75f68f7b82ac971e65b11d03fc4c1b99647))
- ignore summary_signature.py ([ddb7e82](https://github.com/thememium/dspy-auto-signature/commit/ddb7e825cd3784e8fe8ced8f8425957352f55d68))
- **deps**: add boto3>=1.43.15 to dev dependencies ([e2d72a4](https://github.com/thememium/dspy-auto-signature/commit/e2d72a41142043a4d151c096288a41e9808ba13c))
- **pyproject**: tidy formatting and rename dev task ([9685020](https://github.com/thememium/dspy-auto-signature/commit/9685020ae1d92ac03c8463eebbac3a929363af66))
- **dspy-auto-signature**: add __init__.py to generator package ([912f970](https://github.com/thememium/dspy-auto-signature/commit/912f97060075ddacd7e482fe26308def891d842d))
- **pyproject.toml**: update project description to reflect DSPy meta-program signature generation ([11553dc](https://github.com/thememium/dspy-auto-signature/commit/11553dc570ea924e0a6711ef08ce63568829799c))

### ✅ Tests

- expand RLM signature generator and type resolver test suite ([ce5f8c7](https://github.com/thememium/dspy-auto-signature/commit/ce5f8c7b7cb91d74850ca20bf9db641c69bb603a))
- **data**: add comprehensive unit tests for profiler, RLM generator, and to_records ([07a9142](https://github.com/thememium/dspy-auto-signature/commit/07a914248e638633bb8b91d84a47feba81bba95a))
- **types**: add unit tests for types module ([3087016](https://github.com/thememium/dspy-auto-signature/commit/3087016dfe4edbfb1b4a3538e729818ce23c3044))
- **type_resolver**: add tests for natural to Python type resolution ([20458e8](https://github.com/thememium/dspy-auto-signature/commit/20458e8fa78932168be4d2e27160b5367d1951d5))
- add tests for SignatureBuilder ([916b783](https://github.com/thememium/dspy-auto-signature/commit/916b783b69d89cd943276c96b34a190af1ff6170))
- add public API tests for dspy_auto_signature ([be8c6e3](https://github.com/thememium/dspy-auto-signature/commit/be8c6e3c1a7b9a658a6a642501123c26d6c15a8c))
- **parser**: add unit tests for string, Vercel and Auto parsers ([efd8bae](https://github.com/thememium/dspy-auto-signature/commit/efd8bae8484ebe01460e47b74dd7611bf6882613))
- **parser**: add comprehensive tests for parser module ([7e8045b](https://github.com/thememium/dspy-auto-signature/commit/7e8045bbbbd1d5947bb4e0c7e293b3c262883537))
- **config**: add tests for Config module ([3ab526c](https://github.com/thememium/dspy-auto-signature/commit/3ab526ca2d1759bca08a27ed0bf27a71a4eb02c6))

### Other Changes

- Merge pull request #2 from thememium/eboswell/refactor/simplified-auto-signature (#2) ([42842e6](https://github.com/thememium/dspy-auto-signature/commit/42842e634b23cb45bcf299b336190eb5d59d66df))
- Merge pull request #1 from thememium/eboswell/feat/adds-dataframe-and-rlm-generated-signatures (#1) ([f38257b](https://github.com/thememium/dspy-auto-signature/commit/f38257bd5bf9d8490eabf3e777799e6a049907bd))

### Contributors

- Edward Boswell <thememium@gmail.com>
