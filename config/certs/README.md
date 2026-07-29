# Mozilla CA bundle

`mozilla-ca-bundle.pem` 是 curl 官方从 Mozilla NSS 根证书库生成的公开 CA bundle，供 Windows OpenSSL 客户端验证公网 HTTPS 服务端证书。它不包含客户端证书或私钥。

更新并校验 bundle：

```powershell
.\tools\update_mozilla_ca_bundle.ps1
```

来源与说明：<https://curl.se/docs/caextract.html>。该 PEM 沿用 Mozilla 源文件的 MPL 2.0 许可。

更新后应重新执行 TLS 单元测试，并对目标云 API 域名进行带 SNI 和 hostname verification 的握手测试。生产部署包需要同时携带 PEM 文件，不能在缺失时切换为 `verify_none`。
