#include "url_parser.h"

#include <gtest/gtest.h>

using agent::net::ParseUrl;
using agent::net::ParsedUrl;
using agent::net::UrlScheme;

TEST(UrlParserTest, HttpsWithDefaultPort) {
    auto r = ParseUrl("https://api.example.com/v1/chat");
    ASSERT_TRUE(r.ok()) << r.status().message();
    auto u = std::move(r).value();
    EXPECT_EQ(u.scheme, UrlScheme::Https);
    EXPECT_EQ(u.host, "api.example.com");
    EXPECT_EQ(u.port, 443u);
    EXPECT_EQ(u.target, "/v1/chat");
}

TEST(UrlParserTest, HttpWithDefaultPort) {
    auto r = ParseUrl("http://example.com/path");
    ASSERT_TRUE(r.ok());
    auto u = std::move(r).value();
    EXPECT_EQ(u.scheme, UrlScheme::Http);
    EXPECT_EQ(u.port, 80u);
}

TEST(UrlParserTest, ExplicitPort) {
    auto r = ParseUrl("http://127.0.0.1:8080/echo");
    ASSERT_TRUE(r.ok());
    auto u = std::move(r).value();
    EXPECT_EQ(u.host, "127.0.0.1");
    EXPECT_EQ(u.port, 8080u);
    EXPECT_EQ(u.target, "/echo");
}

TEST(UrlParserTest, NoPathDefaultsToRoot) {
    auto r = ParseUrl("https://api.example.com");
    ASSERT_TRUE(r.ok());
    auto u = std::move(r).value();
    EXPECT_EQ(u.target, "/");
}

TEST(UrlParserTest, QueryStringStaysInTarget) {
    auto r = ParseUrl("https://api.example.com/search?q=hi&limit=5");
    ASSERT_TRUE(r.ok());
    auto u = std::move(r).value();
    EXPECT_EQ(u.target, "/search?q=hi&limit=5");
}

TEST(UrlParserTest, RejectsMissingScheme) {
    auto r = ParseUrl("api.example.com/v1");
    EXPECT_FALSE(r.ok());
    EXPECT_EQ(r.status().code(), core::ErrorCode::InvalidArgument);
}

TEST(UrlParserTest, RejectsUnsupportedScheme) {
    auto r = ParseUrl("ftp://example.com/file");
    EXPECT_FALSE(r.ok());
}

TEST(UrlParserTest, RejectsEmptyHost) {
    auto r = ParseUrl("https:///path");
    EXPECT_FALSE(r.ok());
}

TEST(UrlParserTest, RejectsInvalidPort) {
    EXPECT_FALSE(ParseUrl("http://h:abc/").ok());
    EXPECT_FALSE(ParseUrl("http://h:99999/").ok());
    EXPECT_FALSE(ParseUrl("http://h:0/").ok());
    EXPECT_FALSE(ParseUrl("http://h:/").ok());
}

TEST(UrlParserTest, RejectsUserinfoAndFragment) {
    EXPECT_FALSE(ParseUrl("https://user:pass@host/p").ok());
    EXPECT_FALSE(ParseUrl("https://host/p#frag").ok());
}

TEST(UrlParserTest, SchemeIsCaseInsensitive) {
    auto r = ParseUrl("HTTPS://example.com/");
    ASSERT_TRUE(r.ok());
    EXPECT_EQ(r.value().scheme, UrlScheme::Https);
}
