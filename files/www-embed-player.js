(function(){/*

 Copyright The Closure Library Authors.
 SPDX-License-Identifier: Apache-2.0
*/
'use strict';var k;function aa(a){var b=0;return function(){return b<a.length?{done:!1,value:a[b++]}:{done:!0}}}
var ba="function"==typeof Object.defineProperties?Object.defineProperty:function(a,b,c){if(a==Array.prototype||a==Object.prototype)return a;a[b]=c.value;return a};
function ca(a){a=["object"==typeof globalThis&&globalThis,a,"object"==typeof window&&window,"object"==typeof self&&self,"object"==typeof global&&global];for(var b=0;b<a.length;++b){var c=a[b];if(c&&c.Math==Math)return c}throw Error("Cannot find global object");}
var da=ca(this);function p(a,b){if(b)a:{var c=da;a=a.split(".");for(var d=0;d<a.length-1;d++){var e=a[d];if(!(e in c))break a;c=c[e]}a=a[a.length-1];d=c[a];b=b(d);b!=d&&null!=b&&ba(c,a,{configurable:!0,writable:!0,value:b})}}
p("Symbol",function(a){function b(f){if(this instanceof b)throw new TypeError("Symbol is not a constructor");return new c(d+(f||"")+"_"+e++,f)}
function c(f,g){this.h=f;ba(this,"description",{configurable:!0,writable:!0,value:g})}
if(a)return a;c.prototype.toString=function(){return this.h};
var d="jscomp_symbol_"+(1E9*Math.random()>>>0)+"_",e=0;return b});
p("Symbol.iterator",function(a){if(a)return a;a=Symbol("Symbol.iterator");for(var b="Array Int8Array Uint8Array Uint8ClampedArray Int16Array Uint16Array Int32Array Uint32Array Float32Array Float64Array".split(" "),c=0;c<b.length;c++){var d=da[b[c]];"function"===typeof d&&"function"!=typeof d.prototype[a]&&ba(d.prototype,a,{configurable:!0,writable:!0,value:function(){return ea(aa(this))}})}return a});
function ea(a){a={next:a};a[Symbol.iterator]=function(){return this};
return a}
function fa(a){return a.raw=a}
function r(a){var b="undefined"!=typeof Symbol&&Symbol.iterator&&a[Symbol.iterator];if(b)return b.call(a);if("number"==typeof a.length)return{next:aa(a)};throw Error(String(a)+" is not an iterable or ArrayLike");}
function ha(a){if(!(a instanceof Array)){a=r(a);for(var b,c=[];!(b=a.next()).done;)c.push(b.value);a=c}return a}
function ja(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
var ka="function"==typeof Object.assign?Object.assign:function(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(d)for(var e in d)ja(d,e)&&(a[e]=d[e])}return a};
p("Object.assign",function(a){return a||ka});
var la="function"==typeof Object.create?Object.create:function(a){function b(){}
b.prototype=a;return new b},ma=function(){function a(){function c(){}
new c;Reflect.construct(c,[],function(){});
return new c instanceof c}
if("undefined"!=typeof Reflect&&Reflect.construct){if(a())return Reflect.construct;var b=Reflect.construct;return function(c,d,e){c=b(c,d);e&&Reflect.setPrototypeOf(c,e.prototype);return c}}return function(c,d,e){void 0===e&&(e=c);
e=la(e.prototype||Object.prototype);return Function.prototype.apply.call(c,e,d)||e}}(),na;
if("function"==typeof Object.setPrototypeOf)na=Object.setPrototypeOf;else{var oa;a:{var pa={a:!0},qa={};try{qa.__proto__=pa;oa=qa.a;break a}catch(a){}oa=!1}na=oa?function(a,b){a.__proto__=b;if(a.__proto__!==b)throw new TypeError(a+" is not extensible");return a}:null}var ra=na;
function v(a,b){a.prototype=la(b.prototype);a.prototype.constructor=a;if(ra)ra(a,b);else for(var c in b)if("prototype"!=c)if(Object.defineProperties){var d=Object.getOwnPropertyDescriptor(b,c);d&&Object.defineProperty(a,c,d)}else a[c]=b[c];a.va=b.prototype}
function ta(){this.v=!1;this.l=null;this.i=void 0;this.h=1;this.m=this.s=0;this.ra=this.j=null}
function ua(a){if(a.v)throw new TypeError("Generator is already running");a.v=!0}
ta.prototype.X=function(a){this.i=a};
function va(a,b){a.j={nd:b,xd:!0};a.h=a.s||a.m}
ta.prototype.return=function(a){this.j={return:a};this.h=this.m};
function w(a,b,c){a.h=c;return{value:b}}
ta.prototype.A=function(a){this.h=a};
function xa(a,b,c){a.s=b;void 0!=c&&(a.m=c)}
function ya(a,b){a.h=b;a.s=0}
function za(a){a.s=0;var b=a.j.nd;a.j=null;return b}
function Aa(a){a.ra=[a.j];a.s=0;a.m=0}
function Ba(a){var b=a.ra.splice(0)[0];(b=a.j=a.j||b)?b.xd?a.h=a.s||a.m:void 0!=b.A&&a.m<b.A?(a.h=b.A,a.j=null):a.h=a.m:a.h=0}
function Ca(a){this.h=new ta;this.i=a}
function Ea(a,b){ua(a.h);var c=a.h.l;if(c)return Fa(a,"return"in c?c["return"]:function(d){return{value:d,done:!0}},b,a.h.return);
a.h.return(b);return Ga(a)}
function Fa(a,b,c,d){try{var e=b.call(a.h.l,c);if(!(e instanceof Object))throw new TypeError("Iterator result "+e+" is not an object");if(!e.done)return a.h.v=!1,e;var f=e.value}catch(g){return a.h.l=null,va(a.h,g),Ga(a)}a.h.l=null;d.call(a.h,f);return Ga(a)}
function Ga(a){for(;a.h.h;)try{var b=a.i(a.h);if(b)return a.h.v=!1,{value:b.value,done:!1}}catch(c){a.h.i=void 0,va(a.h,c)}a.h.v=!1;if(a.h.j){b=a.h.j;a.h.j=null;if(b.xd)throw b.nd;return{value:b.return,done:!0}}return{value:void 0,done:!0}}
function Ha(a){this.next=function(b){ua(a.h);a.h.l?b=Fa(a,a.h.l.next,b,a.h.X):(a.h.X(b),b=Ga(a));return b};
this.throw=function(b){ua(a.h);a.h.l?b=Fa(a,a.h.l["throw"],b,a.h.X):(va(a.h,b),b=Ga(a));return b};
this.return=function(b){return Ea(a,b)};
this[Symbol.iterator]=function(){return this}}
function Ia(a){function b(d){return a.next(d)}
function c(d){return a.throw(d)}
return new Promise(function(d,e){function f(g){g.done?d(g.value):Promise.resolve(g.value).then(b,c).then(f,e)}
f(a.next())})}
function x(a){return Ia(new Ha(new Ca(a)))}
function Ja(){for(var a=Number(this),b=[],c=a;c<arguments.length;c++)b[c-a]=arguments[c];return b}
p("Reflect",function(a){return a?a:{}});
p("Reflect.construct",function(){return ma});
p("Reflect.setPrototypeOf",function(a){return a?a:ra?function(b,c){try{return ra(b,c),!0}catch(d){return!1}}:null});
p("Promise",function(a){function b(g){this.h=0;this.j=void 0;this.i=[];this.v=!1;var h=this.l();try{g(h.resolve,h.reject)}catch(l){h.reject(l)}}
function c(){this.h=null}
function d(g){return g instanceof b?g:new b(function(h){h(g)})}
if(a)return a;c.prototype.i=function(g){if(null==this.h){this.h=[];var h=this;this.j(function(){h.m()})}this.h.push(g)};
var e=da.setTimeout;c.prototype.j=function(g){e(g,0)};
c.prototype.m=function(){for(;this.h&&this.h.length;){var g=this.h;this.h=[];for(var h=0;h<g.length;++h){var l=g[h];g[h]=null;try{l()}catch(m){this.l(m)}}}this.h=null};
c.prototype.l=function(g){this.j(function(){throw g;})};
b.prototype.l=function(){function g(m){return function(n){l||(l=!0,m.call(h,n))}}
var h=this,l=!1;return{resolve:g(this.T),reject:g(this.m)}};
b.prototype.T=function(g){if(g===this)this.m(new TypeError("A Promise cannot resolve to itself"));else if(g instanceof b)this.Y(g);else{a:switch(typeof g){case "object":var h=null!=g;break a;case "function":h=!0;break a;default:h=!1}h?this.M(g):this.s(g)}};
b.prototype.M=function(g){var h=void 0;try{h=g.then}catch(l){this.m(l);return}"function"==typeof h?this.aa(h,g):this.s(g)};
b.prototype.m=function(g){this.X(2,g)};
b.prototype.s=function(g){this.X(1,g)};
b.prototype.X=function(g,h){if(0!=this.h)throw Error("Cannot settle("+g+", "+h+"): Promise already settled in state"+this.h);this.h=g;this.j=h;2===this.h&&this.V();this.ra()};
b.prototype.V=function(){var g=this;e(function(){if(g.H()){var h=da.console;"undefined"!==typeof h&&h.error(g.j)}},1)};
b.prototype.H=function(){if(this.v)return!1;var g=da.CustomEvent,h=da.Event,l=da.dispatchEvent;if("undefined"===typeof l)return!0;"function"===typeof g?g=new g("unhandledrejection",{cancelable:!0}):"function"===typeof h?g=new h("unhandledrejection",{cancelable:!0}):(g=da.document.createEvent("CustomEvent"),g.initCustomEvent("unhandledrejection",!1,!0,g));g.promise=this;g.reason=this.j;return l(g)};
b.prototype.ra=function(){if(null!=this.i){for(var g=0;g<this.i.length;++g)f.i(this.i[g]);this.i=null}};
var f=new c;b.prototype.Y=function(g){var h=this.l();g.cc(h.resolve,h.reject)};
b.prototype.aa=function(g,h){var l=this.l();try{g.call(h,l.resolve,l.reject)}catch(m){l.reject(m)}};
b.prototype.then=function(g,h){function l(u,t){return"function"==typeof u?function(A){try{m(u(A))}catch(C){n(C)}}:t}
var m,n,q=new b(function(u,t){m=u;n=t});
this.cc(l(g,m),l(h,n));return q};
b.prototype.catch=function(g){return this.then(void 0,g)};
b.prototype.cc=function(g,h){function l(){switch(m.h){case 1:g(m.j);break;case 2:h(m.j);break;default:throw Error("Unexpected state: "+m.h);}}
var m=this;null==this.i?f.i(l):this.i.push(l);this.v=!0};
b.resolve=d;b.reject=function(g){return new b(function(h,l){l(g)})};
b.race=function(g){return new b(function(h,l){for(var m=r(g),n=m.next();!n.done;n=m.next())d(n.value).cc(h,l)})};
b.all=function(g){var h=r(g),l=h.next();return l.done?d([]):new b(function(m,n){function q(A){return function(C){u[A]=C;t--;0==t&&m(u)}}
var u=[],t=0;do u.push(void 0),t++,d(l.value).cc(q(u.length-1),n),l=h.next();while(!l.done)})};
return b});
p("WeakMap",function(a){function b(l){this.h=(h+=Math.random()+1).toString();if(l){l=r(l);for(var m;!(m=l.next()).done;)m=m.value,this.set(m[0],m[1])}}
function c(){}
function d(l){var m=typeof l;return"object"===m&&null!==l||"function"===m}
function e(l){if(!ja(l,g)){var m=new c;ba(l,g,{value:m})}}
function f(l){var m=Object[l];m&&(Object[l]=function(n){if(n instanceof c)return n;Object.isExtensible(n)&&e(n);return m(n)})}
if(function(){if(!a||!Object.seal)return!1;try{var l=Object.seal({}),m=Object.seal({}),n=new a([[l,2],[m,3]]);if(2!=n.get(l)||3!=n.get(m))return!1;n.delete(l);n.set(m,4);return!n.has(l)&&4==n.get(m)}catch(q){return!1}}())return a;
var g="$jscomp_hidden_"+Math.random();f("freeze");f("preventExtensions");f("seal");var h=0;b.prototype.set=function(l,m){if(!d(l))throw Error("Invalid WeakMap key");e(l);if(!ja(l,g))throw Error("WeakMap key fail: "+l);l[g][this.h]=m;return this};
b.prototype.get=function(l){return d(l)&&ja(l,g)?l[g][this.h]:void 0};
b.prototype.has=function(l){return d(l)&&ja(l,g)&&ja(l[g],this.h)};
b.prototype.delete=function(l){return d(l)&&ja(l,g)&&ja(l[g],this.h)?delete l[g][this.h]:!1};
return b});
p("Map",function(a){function b(){var h={};return h.previous=h.next=h.head=h}
function c(h,l){var m=h.h;return ea(function(){if(m){for(;m.head!=h.h;)m=m.previous;for(;m.next!=m.head;)return m=m.next,{done:!1,value:l(m)};m=null}return{done:!0,value:void 0}})}
function d(h,l){var m=l&&typeof l;"object"==m||"function"==m?f.has(l)?m=f.get(l):(m=""+ ++g,f.set(l,m)):m="p_"+l;var n=h.data_[m];if(n&&ja(h.data_,m))for(h=0;h<n.length;h++){var q=n[h];if(l!==l&&q.key!==q.key||l===q.key)return{id:m,list:n,index:h,entry:q}}return{id:m,list:n,index:-1,entry:void 0}}
function e(h){this.data_={};this.h=b();this.size=0;if(h){h=r(h);for(var l;!(l=h.next()).done;)l=l.value,this.set(l[0],l[1])}}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var h=Object.seal({x:4}),l=new a(r([[h,"s"]]));if("s"!=l.get(h)||1!=l.size||l.get({x:4})||l.set({x:4},"t")!=l||2!=l.size)return!1;var m=l.entries(),n=m.next();if(n.done||n.value[0]!=h||"s"!=n.value[1])return!1;n=m.next();return n.done||4!=n.value[0].x||"t"!=n.value[1]||!m.next().done?!1:!0}catch(q){return!1}}())return a;
var f=new WeakMap;e.prototype.set=function(h,l){h=0===h?0:h;var m=d(this,h);m.list||(m.list=this.data_[m.id]=[]);m.entry?m.entry.value=l:(m.entry={next:this.h,previous:this.h.previous,head:this.h,key:h,value:l},m.list.push(m.entry),this.h.previous.next=m.entry,this.h.previous=m.entry,this.size++);return this};
e.prototype.delete=function(h){h=d(this,h);return h.entry&&h.list?(h.list.splice(h.index,1),h.list.length||delete this.data_[h.id],h.entry.previous.next=h.entry.next,h.entry.next.previous=h.entry.previous,h.entry.head=null,this.size--,!0):!1};
e.prototype.clear=function(){this.data_={};this.h=this.h.previous=b();this.size=0};
e.prototype.has=function(h){return!!d(this,h).entry};
e.prototype.get=function(h){return(h=d(this,h).entry)&&h.value};
e.prototype.entries=function(){return c(this,function(h){return[h.key,h.value]})};
e.prototype.keys=function(){return c(this,function(h){return h.key})};
e.prototype.values=function(){return c(this,function(h){return h.value})};
e.prototype.forEach=function(h,l){for(var m=this.entries(),n;!(n=m.next()).done;)n=n.value,h.call(l,n[1],n[0],this)};
e.prototype[Symbol.iterator]=e.prototype.entries;var g=0;return e});
function Ka(a,b,c){if(null==a)throw new TypeError("The 'this' value for String.prototype."+c+" must not be null or undefined");if(b instanceof RegExp)throw new TypeError("First argument to String.prototype."+c+" must not be a regular expression");return a+""}
p("String.prototype.endsWith",function(a){return a?a:function(b,c){var d=Ka(this,b,"endsWith");b+="";void 0===c&&(c=d.length);c=Math.max(0,Math.min(c|0,d.length));for(var e=b.length;0<e&&0<c;)if(d[--c]!=b[--e])return!1;return 0>=e}});
p("Array.prototype.find",function(a){return a?a:function(b,c){a:{var d=this;d instanceof String&&(d=String(d));for(var e=d.length,f=0;f<e;f++){var g=d[f];if(b.call(c,g,f,d)){b=g;break a}}b=void 0}return b}});
p("String.prototype.startsWith",function(a){return a?a:function(b,c){var d=Ka(this,b,"startsWith");b+="";var e=d.length,f=b.length;c=Math.max(0,Math.min(c|0,d.length));for(var g=0;g<f&&c<e;)if(d[c++]!=b[g++])return!1;return g>=f}});
p("Number.isFinite",function(a){return a?a:function(b){return"number"!==typeof b?!1:!isNaN(b)&&Infinity!==b&&-Infinity!==b}});
p("Number.MAX_SAFE_INTEGER",function(){return 9007199254740991});
p("Number.isNaN",function(a){return a?a:function(b){return"number"===typeof b&&isNaN(b)}});
function La(a,b){a instanceof String&&(a+="");var c=0,d=!1,e={next:function(){if(!d&&c<a.length){var f=c++;return{value:b(f,a[f]),done:!1}}d=!0;return{done:!0,value:void 0}}};
e[Symbol.iterator]=function(){return e};
return e}
p("Array.prototype.entries",function(a){return a?a:function(){return La(this,function(b,c){return[b,c]})}});
p("Array.prototype.keys",function(a){return a?a:function(){return La(this,function(b){return b})}});
p("Array.from",function(a){return a?a:function(b,c,d){c=null!=c?c:function(h){return h};
var e=[],f="undefined"!=typeof Symbol&&Symbol.iterator&&b[Symbol.iterator];if("function"==typeof f){b=f.call(b);for(var g=0;!(f=b.next()).done;)e.push(c.call(d,f.value,g++))}else for(f=b.length,g=0;g<f;g++)e.push(c.call(d,b[g],g));return e}});
p("Number.isInteger",function(a){return a?a:function(b){return Number.isFinite(b)?b===Math.floor(b):!1}});
p("Array.prototype.values",function(a){return a?a:function(){return La(this,function(b,c){return c})}});
p("Object.setPrototypeOf",function(a){return a||ra});
p("Set",function(a){function b(c){this.h=new Map;if(c){c=r(c);for(var d;!(d=c.next()).done;)this.add(d.value)}this.size=this.h.size}
if(function(){if(!a||"function"!=typeof a||!a.prototype.entries||"function"!=typeof Object.seal)return!1;try{var c=Object.seal({x:4}),d=new a(r([c]));if(!d.has(c)||1!=d.size||d.add(c)!=d||1!=d.size||d.add({x:4})!=d||2!=d.size)return!1;var e=d.entries(),f=e.next();if(f.done||f.value[0]!=c||f.value[1]!=c)return!1;f=e.next();return f.done||f.value[0]==c||4!=f.value[0].x||f.value[1]!=f.value[0]?!1:e.next().done}catch(g){return!1}}())return a;
b.prototype.add=function(c){c=0===c?0:c;this.h.set(c,c);this.size=this.h.size;return this};
b.prototype.delete=function(c){c=this.h.delete(c);this.size=this.h.size;return c};
b.prototype.clear=function(){this.h.clear();this.size=0};
b.prototype.has=function(c){return this.h.has(c)};
b.prototype.entries=function(){return this.h.entries()};
b.prototype.values=function(){return this.h.values()};
b.prototype.keys=b.prototype.values;b.prototype[Symbol.iterator]=b.prototype.values;b.prototype.forEach=function(c,d){var e=this;this.h.forEach(function(f){return c.call(d,f,f,e)})};
return b});
p("Object.entries",function(a){return a?a:function(b){var c=[],d;for(d in b)ja(b,d)&&c.push([d,b[d]]);return c}});
p("Object.is",function(a){return a?a:function(b,c){return b===c?0!==b||1/b===1/c:b!==b&&c!==c}});
p("Array.prototype.includes",function(a){return a?a:function(b,c){var d=this;d instanceof String&&(d=String(d));var e=d.length;c=c||0;for(0>c&&(c=Math.max(c+e,0));c<e;c++){var f=d[c];if(f===b||Object.is(f,b))return!0}return!1}});
p("String.prototype.includes",function(a){return a?a:function(b,c){return-1!==Ka(this,b,"includes").indexOf(b,c||0)}});
p("globalThis",function(a){return a||da});
p("Object.values",function(a){return a?a:function(b){var c=[],d;for(d in b)ja(b,d)&&c.push(b[d]);return c}});
var Ma=Ma||{},y=this||self;function z(a,b,c){a=a.split(".");c=c||y;a[0]in c||"undefined"==typeof c.execScript||c.execScript("var "+a[0]);for(var d;a.length&&(d=a.shift());)a.length||void 0===b?c[d]&&c[d]!==Object.prototype[d]?c=c[d]:c=c[d]={}:c[d]=b}
function B(a,b){a=a.split(".");b=b||y;for(var c=0;c<a.length;c++)if(b=b[a[c]],null==b)return null;return b}
function Na(a){var b=typeof a;return"object"!=b?b:a?Array.isArray(a)?"array":b:"null"}
function Oa(a){var b=Na(a);return"array"==b||"object"==b&&"number"==typeof a.length}
function Pa(a){var b=typeof a;return"object"==b&&null!=a||"function"==b}
function Ra(a){return Object.prototype.hasOwnProperty.call(a,Sa)&&a[Sa]||(a[Sa]=++Ta)}
var Sa="closure_uid_"+(1E9*Math.random()>>>0),Ta=0;function Ua(a,b,c){return a.call.apply(a.bind,arguments)}
function Va(a,b,c){if(!a)throw Error();if(2<arguments.length){var d=Array.prototype.slice.call(arguments,2);return function(){var e=Array.prototype.slice.call(arguments);Array.prototype.unshift.apply(e,d);return a.apply(b,e)}}return function(){return a.apply(b,arguments)}}
function Wa(a,b,c){Function.prototype.bind&&-1!=Function.prototype.bind.toString().indexOf("native code")?Wa=Ua:Wa=Va;return Wa.apply(null,arguments)}
function Xa(a,b){var c=Array.prototype.slice.call(arguments,1);return function(){var d=c.slice();d.push.apply(d,arguments);return a.apply(this,d)}}
function Ya(a,b){function c(){}
c.prototype=b.prototype;a.va=b.prototype;a.prototype=new c;a.prototype.constructor=a;a.pg=function(d,e,f){for(var g=Array(arguments.length-2),h=2;h<arguments.length;h++)g[h-2]=arguments[h];return b.prototype[e].apply(d,g)}}
function Za(a){return a}
;function $a(a,b){if(Error.captureStackTrace)Error.captureStackTrace(this,$a);else{var c=Error().stack;c&&(this.stack=c)}a&&(this.message=String(a));void 0!==b&&(this.cause=b)}
Ya($a,Error);$a.prototype.name="CustomError";function ab(a){a=a.url;var b=/[?&]dsh=1(&|$)/.test(a);this.j=!b&&/[?&]ae=1(&|$)/.test(a);this.l=!b&&/[?&]ae=2(&|$)/.test(a);if((this.h=/[?&]adurl=([^&]*)/.exec(a))&&this.h[1]){try{var c=decodeURIComponent(this.h[1])}catch(d){c=null}this.i=c}}
;function cb(){}
function db(a){var b=!1,c;return function(){b||(c=a(),b=!0);return c}}
;var eb=Array.prototype.indexOf?function(a,b){return Array.prototype.indexOf.call(a,b,void 0)}:function(a,b){if("string"===typeof a)return"string"!==typeof b||1!=b.length?-1:a.indexOf(b,0);
for(var c=0;c<a.length;c++)if(c in a&&a[c]===b)return c;return-1},fb=Array.prototype.forEach?function(a,b,c){Array.prototype.forEach.call(a,b,c)}:function(a,b,c){for(var d=a.length,e="string"===typeof a?a.split(""):a,f=0;f<d;f++)f in e&&b.call(c,e[f],f,a)},gb=Array.prototype.filter?function(a,b){return Array.prototype.filter.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=[],e=0,f="string"===typeof a?a.split(""):a,g=0;g<c;g++)if(g in f){var h=f[g];
b.call(void 0,h,g,a)&&(d[e++]=h)}return d},hb=Array.prototype.map?function(a,b){return Array.prototype.map.call(a,b,void 0)}:function(a,b){for(var c=a.length,d=Array(c),e="string"===typeof a?a.split(""):a,f=0;f<c;f++)f in e&&(d[f]=b.call(void 0,e[f],f,a));
return d},ib=Array.prototype.reduce?function(a,b,c){return Array.prototype.reduce.call(a,b,c)}:function(a,b,c){var d=c;
fb(a,function(e,f){d=b.call(void 0,d,e,f,a)});
return d};
function jb(a,b){a:{for(var c=a.length,d="string"===typeof a?a.split(""):a,e=0;e<c;e++)if(e in d&&b.call(void 0,d[e],e,a)){b=e;break a}b=-1}return 0>b?null:"string"===typeof a?a.charAt(b):a[b]}
function kb(a,b){b=eb(a,b);var c;(c=0<=b)&&Array.prototype.splice.call(a,b,1);return c}
function lb(a,b){for(var c=1;c<arguments.length;c++){var d=arguments[c];if(Oa(d)){var e=a.length||0,f=d.length||0;a.length=e+f;for(var g=0;g<f;g++)a[e+g]=d[g]}else a.push(d)}}
;function mb(a,b){for(var c in a)b.call(void 0,a[c],c,a)}
function nb(a){var b=ob,c;for(c in b)if(a.call(void 0,b[c],c,b))return c}
function pb(a){for(var b in a)return!1;return!0}
function qb(a,b){if(null!==a&&b in a)throw Error('The object already contains the key "'+b+'"');a[b]=!0}
function rb(a){return null!==a&&"privembed"in a?a.privembed:!1}
function sb(a,b){for(var c in a)if(!(c in b)||a[c]!==b[c])return!1;for(var d in b)if(!(d in a))return!1;return!0}
function tb(a){var b={},c;for(c in a)b[c]=a[c];return b}
function ub(a){if(!a||"object"!==typeof a)return a;if("function"===typeof a.clone)return a.clone();if("undefined"!==typeof Map&&a instanceof Map)return new Map(a);if("undefined"!==typeof Set&&a instanceof Set)return new Set(a);if(a instanceof Date)return new Date(a.getTime());var b=Array.isArray(a)?[]:"function"!==typeof ArrayBuffer||"function"!==typeof ArrayBuffer.isView||!ArrayBuffer.isView(a)||a instanceof DataView?{}:new a.constructor(a.length),c;for(c in a)b[c]=ub(a[c]);return b}
var vb="constructor hasOwnProperty isPrototypeOf propertyIsEnumerable toLocaleString toString valueOf".split(" ");function wb(a,b){for(var c,d,e=1;e<arguments.length;e++){d=arguments[e];for(c in d)a[c]=d[c];for(var f=0;f<vb.length;f++)c=vb[f],Object.prototype.hasOwnProperty.call(d,c)&&(a[c]=d[c])}}
;var xb;function yb(){if(void 0===xb){var a=null,b=y.trustedTypes;if(b&&b.createPolicy){try{a=b.createPolicy("goog#html",{createHTML:Za,createScript:Za,createScriptURL:Za})}catch(c){y.console&&y.console.error(c.message)}xb=a}else xb=a}return xb}
;function zb(a,b){this.j=a===Ab&&b||""}
zb.prototype.i=!0;zb.prototype.h=function(){return this.j};
function Bb(a){return new zb(Ab,a)}
var Ab={};Bb("");var Cb={};function Db(a,b){this.j=b===Cb?a:"";this.i=!0}
Db.prototype.toString=function(){return this.j.toString()};
Db.prototype.h=function(){return this.j.toString()};function Eb(a,b){this.j=b===Fb?a:""}
Eb.prototype.toString=function(){return this.j+""};
Eb.prototype.i=!0;Eb.prototype.h=function(){return this.j.toString()};
function Gb(a){if(a instanceof Eb&&a.constructor===Eb)return a.j;Na(a);return"type_error:TrustedResourceUrl"}
var Fb={};function Hb(a){var b=yb();a=b?b.createScriptURL(a):a;return new Eb(a,Fb)}
;var Ib=String.prototype.trim?function(a){return a.trim()}:function(a){return/^[\s\xa0]*([\s\S]*?)[\s\xa0]*$/.exec(a)[1]};
function Jb(a,b){return a<b?-1:a>b?1:0}
;function Kb(a,b){this.j=b===Lb?a:""}
Kb.prototype.toString=function(){return this.j.toString()};
Kb.prototype.i=!0;Kb.prototype.h=function(){return this.j.toString()};
function Mb(a){if(a instanceof Kb&&a.constructor===Kb)return a.j;Na(a);return"type_error:SafeUrl"}
var Nb;try{new URL("s://g"),Nb=!0}catch(a){Nb=!1}var Ob=Nb,Lb={},Pb=new Kb("about:invalid#zClosurez",Lb);var Qb,Rb=B("CLOSURE_FLAGS"),Ub=Rb&&Rb[610401301];Qb=null!=Ub?Ub:!1;function Vb(){var a=y.navigator;return a&&(a=a.userAgent)?a:""}
var Wb,Xb=y.navigator;Wb=Xb?Xb.userAgentData||null:null;function Yb(a){return Qb?Wb?Wb.brands.some(function(b){return(b=b.brand)&&-1!=b.indexOf(a)}):!1:!1}
function D(a){return-1!=Vb().indexOf(a)}
;function Zb(){return Qb?!!Wb&&0<Wb.brands.length:!1}
function $b(){return Zb()?!1:D("Opera")}
function ac(){return Zb()?!1:D("Trident")||D("MSIE")}
function bc(){return Zb()?!1:D("Edge")}
function cc(){return Zb()?Yb("Microsoft Edge"):D("Edg/")}
function dc(){return D("Firefox")||D("FxiOS")}
function ec(){return D("Safari")&&!(fc()||(Zb()?0:D("Coast"))||$b()||bc()||cc()||(Zb()?Yb("Opera"):D("OPR"))||dc()||D("Silk")||D("Android"))}
function fc(){return Zb()?Yb("Chromium"):(D("Chrome")||D("CriOS"))&&!bc()||D("Silk")}
function gc(){return D("Android")&&!(fc()||dc()||$b()||D("Silk"))}
function hc(a){var b={};a.forEach(function(c){b[c[0]]=c[1]});
return function(c){return b[c.find(function(d){return d in b})]||""}}
function ic(a){var b=Vb();if("Internet Explorer"===a){if(ac())if((a=/rv: *([\d\.]*)/.exec(b))&&a[1])b=a[1];else{a="";var c=/MSIE +([\d\.]+)/.exec(b);if(c&&c[1])if(b=/Trident\/(\d.\d)/.exec(b),"7.0"==c[1])if(b&&b[1])switch(b[1]){case "4.0":a="8.0";break;case "5.0":a="9.0";break;case "6.0":a="10.0";break;case "7.0":a="11.0"}else a="7.0";else a=c[1];b=a}else b="";return b}var d=RegExp("([A-Z][\\w ]+)/([^\\s]+)\\s*(?:\\((.*?)\\))?","g");c=[];for(var e;e=d.exec(b);)c.push([e[1],e[2],e[3]||void 0]);b=hc(c);
switch(a){case "Opera":if($b())return b(["Version","Opera"]);if(Zb()?Yb("Opera"):D("OPR"))return b(["OPR"]);break;case "Microsoft Edge":if(bc())return b(["Edge"]);if(cc())return b(["Edg"]);break;case "Chromium":if(fc())return b(["Chrome","CriOS","HeadlessChrome"])}return"Firefox"===a&&dc()||"Safari"===a&&ec()||"Android Browser"===a&&gc()||"Silk"===a&&D("Silk")?(b=c[2])&&b[1]||"":""}
function jc(a){if(Zb()&&"Silk"!==a){var b=Wb.brands.find(function(c){return c.brand===a});
if(!b||!b.version)return NaN;b=b.version.split(".")}else{b=ic(a);if(""===b)return NaN;b=b.split(".")}return 0===b.length?NaN:Number(b[0])}
;var kc={};function lc(a){this.j=kc===kc?a:"";this.i=!0}
lc.prototype.h=function(){return this.j.toString()};
lc.prototype.toString=function(){return this.j.toString()};function mc(a,b){if(!(b instanceof Kb||b instanceof Kb)){b="object"==typeof b&&b.i?b.h():String(b);b:{var c=b;if(Ob){try{var d=new URL(c)}catch(e){c="https:";break b}c=d.protocol}else c:{d=document.createElement("a");try{d.href=c}catch(e){c=void 0;break c}c=d.protocol;c=":"===c||""===c?"https:":c}}"javascript:"!==c||(b="about:invalid#zClosurez");b=new Kb(b,Lb)}a.href=Mb(b)}
function nc(a,b){a.rel="stylesheet";a.href=Gb(b).toString();(b=oc('style[nonce],link[rel="stylesheet"][nonce]',a.ownerDocument&&a.ownerDocument.defaultView))&&a.setAttribute("nonce",b)}
function qc(){return oc("script[nonce]")}
var rc=/^[\w+/_-]+[=]{0,2}$/;function oc(a,b){b=(b||y).document;return b.querySelector?(a=b.querySelector(a))&&(a=a.nonce||a.getAttribute("nonce"))&&rc.test(a)?a:"":""}
;function sc(a){for(var b=0,c=0;c<a.length;++c)b=31*b+a.charCodeAt(c)>>>0;return b}
;var tc=RegExp("^(?:([^:/?#.]+):)?(?://(?:([^\\\\/?#]*)@)?([^\\\\/?#]*?)(?::([0-9]+))?(?=[\\\\/?#]|$))?([^?#]+)?(?:\\?([^#]*))?(?:#([\\s\\S]*))?$");function uc(a){return a?decodeURI(a):a}
function vc(a,b){return b.match(tc)[a]||null}
function wc(a){return uc(vc(3,a))}
function xc(a){var b=a.match(tc);a=b[5];var c=b[6];b=b[7];var d="";a&&(d+=a);c&&(d+="?"+c);b&&(d+="#"+b);return d}
function yc(a,b){if(!b)return a;var c=a.indexOf("#");0>c&&(c=a.length);var d=a.indexOf("?");if(0>d||d>c){d=c;var e=""}else e=a.substring(d+1,c);a=[a.slice(0,d),e,a.slice(c)];c=a[1];a[1]=b?c?c+"&"+b:b:c;return a[0]+(a[1]?"?"+a[1]:"")+a[2]}
function zc(a,b,c){if(Array.isArray(b))for(var d=0;d<b.length;d++)zc(a,String(b[d]),c);else null!=b&&c.push(a+(""===b?"":"="+encodeURIComponent(String(b))))}
function Ac(a,b){var c=[];for(b=b||0;b<a.length;b+=2)zc(a[b],a[b+1],c);return c.join("&")}
function Bc(a){var b=[],c;for(c in a)zc(c,a[c],b);return b.join("&")}
function Cc(a,b){var c=2==arguments.length?Ac(arguments[1],0):Ac(arguments,1);return yc(a,c)}
function Dc(a,b){b=Bc(b);return yc(a,b)}
function Ec(a,b,c){c=null!=c?"="+encodeURIComponent(String(c)):"";return yc(a,b+c)}
function Fc(a,b,c,d){for(var e=c.length;0<=(b=a.indexOf(c,b))&&b<d;){var f=a.charCodeAt(b-1);if(38==f||63==f)if(f=a.charCodeAt(b+e),!f||61==f||38==f||35==f)return b;b+=e+1}return-1}
var Gc=/#|$/,Hc=/[?&]($|#)/;function Ic(a,b){for(var c=a.search(Gc),d=0,e,f=[];0<=(e=Fc(a,d,b,c));)f.push(a.substring(d,e)),d=Math.min(a.indexOf("&",e)+1||c,c);f.push(a.slice(d));return f.join("").replace(Hc,"$1")}
;function Jc(a){y.setTimeout(function(){throw a;},0)}
;function Kc(a){for(var b=[],c=0,d=0;d<a.length;d++){var e=a.charCodeAt(d);128>e?b[c++]=e:(2048>e?b[c++]=e>>6|192:(55296==(e&64512)&&d+1<a.length&&56320==(a.charCodeAt(d+1)&64512)?(e=65536+((e&1023)<<10)+(a.charCodeAt(++d)&1023),b[c++]=e>>18|240,b[c++]=e>>12&63|128):b[c++]=e>>12|224,b[c++]=e>>6&63|128),b[c++]=e&63|128)}return b}
;function Lc(){return Qb?!!Wb&&!!Wb.platform:!1}
function Mc(){return Lc()?"Android"===Wb.platform:D("Android")}
function Nc(){return D("iPhone")&&!D("iPod")&&!D("iPad")}
function Oc(){var a=Vb(),b="";if(Lc()?"Windows"===Wb.platform:D("Windows"))b=/Windows (?:NT|Phone) ([0-9.]+)/,b=(a=b.exec(a))?a[1]:"0.0";else if(Nc()||D("iPad")||D("iPod"))b=/(?:iPhone|iPod|iPad|CPU)\s+OS\s+(\S+)/,b=(a=b.exec(a))&&a[1].replace(/_/g,".");else if(Lc()?"macOS"===Wb.platform:D("Macintosh"))b=/Mac OS X ([0-9_.]+)/,b=(a=b.exec(a))?a[1].replace(/_/g,"."):"10";else if(-1!=Vb().toLowerCase().indexOf("kaios"))b=/(?:KaiOS)\/(\S+)/i,b=(a=b.exec(a))&&a[1];else if(Mc())b=/Android\s+([^\);]+)(\)|;)/,
b=(a=b.exec(a))&&a[1];else if(Lc()?"Chrome OS"===Wb.platform:D("CrOS"))b=/(?:CrOS\s+(?:i686|x86_64)\s+([0-9.]+))/,b=(a=b.exec(a))&&a[1];a=0;b=Ib(String(b||"")).split(".");for(var c=Ib("12").split("."),d=Math.max(b.length,c.length),e=0;0==a&&e<d;e++){var f=b[e]||"",g=c[e]||"";do{f=/(\d*)(\D*)(.*)/.exec(f)||["","","",""];g=/(\d*)(\D*)(.*)/.exec(g)||["","","",""];if(0==f[0].length&&0==g[0].length)break;a=Jb(0==f[1].length?0:parseInt(f[1],10),0==g[1].length?0:parseInt(g[1],10))||Jb(0==f[2].length,0==
g[2].length)||Jb(f[2],g[2]);f=f[3];g=g[3]}while(0==a)}}
;function Uc(a){Uc[" "](a);return a}
Uc[" "]=function(){};var Vc=$b(),Wc=ac(),Xc=D("Edge"),Yc=D("Gecko")&&!(-1!=Vb().toLowerCase().indexOf("webkit")&&!D("Edge"))&&!(D("Trident")||D("MSIE"))&&!D("Edge"),Zc=-1!=Vb().toLowerCase().indexOf("webkit")&&!D("Edge"),$c=Mc();function ad(){var a=y.document;return a?a.documentMode:void 0}
var bd;a:{var cd="",dd=function(){var a=Vb();if(Yc)return/rv:([^\);]+)(\)|;)/.exec(a);if(Xc)return/Edge\/([\d\.]+)/.exec(a);if(Wc)return/\b(?:MSIE|rv)[: ]([^\);]+)(\)|;)/.exec(a);if(Zc)return/WebKit\/(\S+)/.exec(a);if(Vc)return/(?:Version)[ \/]?(\S+)/.exec(a)}();
dd&&(cd=dd?dd[1]:"");if(Wc){var ed=ad();if(null!=ed&&ed>parseFloat(cd)){bd=String(ed);break a}}bd=cd}var fd=bd,gd;if(y.document&&Wc){var hd=ad();gd=hd?hd:parseInt(fd,10)||void 0}else gd=void 0;var id=gd;var jd=Nc()||D("iPod"),kd=D("iPad");gc();fc();var ld=ec()&&!(Nc()||D("iPad")||D("iPod"));var md={},nd=null;function od(a,b){Oa(a);void 0===b&&(b=0);pd();b=md[b];for(var c=Array(Math.floor(a.length/3)),d=b[64]||"",e=0,f=0;e<a.length-2;e+=3){var g=a[e],h=a[e+1],l=a[e+2],m=b[g>>2];g=b[(g&3)<<4|h>>4];h=b[(h&15)<<2|l>>6];l=b[l&63];c[f++]=""+m+g+h+l}m=0;l=d;switch(a.length-e){case 2:m=a[e+1],l=b[(m&15)<<2]||d;case 1:a=a[e],c[f]=""+b[a>>2]+b[(a&3)<<4|m>>4]+l+d}return c.join("")}
function qd(a){var b=a.length,c=3*b/4;c%3?c=Math.floor(c):-1!="=.".indexOf(a[b-1])&&(c=-1!="=.".indexOf(a[b-2])?c-2:c-1);var d=new Uint8Array(c),e=0;rd(a,function(f){d[e++]=f});
return e!==c?d.subarray(0,e):d}
function rd(a,b){function c(l){for(;d<a.length;){var m=a.charAt(d++),n=nd[m];if(null!=n)return n;if(!/^[\s\xa0]*$/.test(m))throw Error("Unknown base64 encoding at char: "+m);}return l}
pd();for(var d=0;;){var e=c(-1),f=c(0),g=c(64),h=c(64);if(64===h&&-1===e)break;b(e<<2|f>>4);64!=g&&(b(f<<4&240|g>>2),64!=h&&b(g<<6&192|h))}}
function pd(){if(!nd){nd={};for(var a="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789".split(""),b=["+/=","+/","-_=","-_.","-_"],c=0;5>c;c++){var d=a.concat(b[c].split(""));md[c]=d;for(var e=0;e<d.length;e++){var f=d[e];void 0===nd[f]&&(nd[f]=e)}}}}
;var sd="undefined"!==typeof Uint8Array,td=!Wc&&"function"===typeof y.btoa;function ud(a){if(!td)return od(a);for(var b="";10240<a.length;)b+=String.fromCharCode.apply(null,a.subarray(0,10240)),a=a.subarray(10240);b+=String.fromCharCode.apply(null,a);return btoa(b)}
var vd=/[-_.]/g,wd={"-":"+",_:"/",".":"="};function xd(a){return wd[a]||""}
function yd(a){return sd&&null!=a&&a instanceof Uint8Array}
var zd={};var Ad;function Bd(a){if(a!==zd)throw Error("illegal external caller");}
function Cd(a,b){Bd(b);this.h=a;if(null!=a&&0===a.length)throw Error("ByteString should be constructed with non-empty values");}
Cd.prototype.sizeBytes=function(){Bd(zd);var a=this.h;if(null!=a&&!yd(a))if("string"===typeof a)if(td){vd.test(a)&&(a=a.replace(vd,xd));a=atob(a);for(var b=new Uint8Array(a.length),c=0;c<a.length;c++)b[c]=a.charCodeAt(c);a=b}else a=qd(a);else Na(a),a=null;return(a=null==a?a:this.h=a)?a.length:0};var Dd="function"===typeof Symbol&&"symbol"===typeof Symbol()?Symbol():void 0;function Ed(a,b){if(Dd)return a[Dd]|=b;if(void 0!==a.Ca)return a.Ca|=b;Object.defineProperties(a,{Ca:{value:b,configurable:!0,writable:!0,enumerable:!1}});return b}
function Fd(a,b){var c=F(a);(c&b)!==b&&(Object.isFrozen(a)&&(a=Array.prototype.slice.call(a)),Gd(a,c|b));return a}
function Hd(a,b){Dd?a[Dd]&&(a[Dd]&=~b):void 0!==a.Ca&&(a.Ca&=~b)}
function F(a){var b;Dd?b=a[Dd]:b=a.Ca;return null==b?0:b}
function Gd(a,b){Dd?a[Dd]=b:void 0!==a.Ca?a.Ca=b:Object.defineProperties(a,{Ca:{value:b,configurable:!0,writable:!0,enumerable:!1}});return a}
function Id(a){Ed(a,1);return a}
function Jd(a,b){Gd(b,(a|0)&-51)}
function Kd(a,b){Gd(b,(a|18)&-41)}
;var Ld={};function Md(a){return null!==a&&"object"===typeof a&&!Array.isArray(a)&&a.constructor===Object}
var Qd,Rd=Object.freeze(Gd([],23));function Sd(a){if(F(a.P)&2)throw Error();}
function Td(a){var b=a.length;(b=b?a[b-1]:void 0)&&Md(b)?b.g=1:(b={},a.push((b.g=1,b)))}
;function Ud(a){if(null!=a&&"number"!==typeof a)throw Error("Value of float/double field must be a number|null|undefined, found "+typeof a+": "+a);return a}
function Vd(a){return a.displayName||a.name||"unknown type name"}
function Wd(a,b){if(!(a instanceof b))throw Error("Expected instanceof "+Vd(b)+" but got "+(a&&Vd(a.constructor)));return a}
function Xd(a,b){var c=F(a),d=c;0===d&&(d|=b&16);d|=b&2;d!==c&&Gd(a,d)}
;function Yd(a){var b=a.j+a.fb;return a.xa||(a.xa=a.P[b]={})}
function Zd(a,b,c){return-1===b?null:b>=a.j?a.xa?a.xa[b]:void 0:c&&a.xa&&(c=a.xa[b],null!=c)?c:a.P[b+a.fb]}
function G(a,b,c,d){Sd(a);return $d(a,b,c,d)}
function $d(a,b,c,d){a.m&&(a.m=void 0);if(b>=a.j||d)return Yd(a)[b]=c,a;a.P[b+a.fb]=c;(c=a.xa)&&b in c&&delete c[b];return a}
function ae(a){return void 0!==be(a,ce,11,!1)}
function de(a,b,c,d,e){var f=Zd(a,b,d);Array.isArray(f)||(f=Rd);var g=F(f);g&1||Id(f);if(e)g&2||Ed(f,18),c&1||Object.freeze(f);else{e=!(c&2);var h=g&2;c&1||!h?e&&g&16&&!h&&Hd(f,16):(f=Id(Array.prototype.slice.call(f)),$d(a,b,f,d))}return f}
function ee(a,b,c,d){Sd(a);(c=fe(a,c))&&c!==b&&null!=d&&$d(a,c,void 0,!1);return $d(a,b,d)}
function fe(a,b){for(var c=0,d=0;d<b.length;d++){var e=b[d];null!=Zd(a,e)&&(0!==c&&$d(a,c,void 0,!1),c=e)}return c}
function be(a,b,c,d){var e=Zd(a,c,d);var f=F(a.P),g=!1;null==e||"object"!==typeof e||(g=Array.isArray(e))||e.Jc!==Ld?g?(Xd(e,f),b=new b(e)):b=void 0:b=e;b!==e&&null!=b&&$d(a,c,b,d);return b}
function ge(a,b,c,d){d=void 0===d?!1:d;b=be(a,b,c,d);if(null==b)return b;if(!(F(a.P)&2)){var e=he(b);e!==b&&(b=e,$d(a,c,b,d))}return b}
function ie(a,b,c,d,e){var f=!!(e&2);a.i||(a.i={});var g=a.i[c],h=de(a,c,3,void 0,f);if(!g){var l=h;g=[];f=!!(e&2);h=!!(F(l)&2);var m=l;!f&&h&&(l=Array.prototype.slice.call(l));var n=e|(h?2:0);e=h;for(var q=0;q<l.length;q++){var u=l[q];var t=b;Array.isArray(u)?(Xd(u,n),u=new t(u)):u=void 0;void 0!==u&&(e=e||!!(2&F(u.P)),g.push(u))}a.i[c]=g;b=l;e=!e;n=F(b);l=n|33;l=e?l|8:l&-9;n!=l&&(Object.isFrozen(b)&&(b=Array.prototype.slice.call(b)),Gd(b,l));l=b;m!==l&&$d(a,c,l);(f||1===d&&h)&&Ed(g,18);(f||1===
d)&&Object.freeze(g);return g}if(3===d)return g;f||((f=Object.isFrozen(g),1!==d||f)?2===d&&f&&(g=Array.prototype.slice.call(g),a.i[c]=g):Object.freeze(g));return g}
function H(a,b,c,d){Sd(a);null!=d?Wd(d,b):d=void 0;return $d(a,c,d)}
function je(a,b,c,d,e){Sd(a);null!=e?Wd(e,b):e=void 0;ee(a,c,d,e)}
function ke(a,b,c,d,e){Sd(a);var f=null==d?Rd:Id([]);if(null!=d){for(var g=!!d.length,h=0;h<d.length;h++){var l=d[h];Wd(l,b);g=g&&!(F(l.P)&2);f[h]=l.P}f=Fd(f,(g?8:0)|1);a.i||(a.i={});a.i[c]=d}else a.i&&(a.i[c]=void 0);return $d(a,c,f,e)}
function le(a,b,c,d){var e=F(a.P);if(e&2)throw Error();e=ie(a,c,b,2,e);c=null!=d?Wd(d,c):new c;a=de(a,b,2,void 0,!1);e.push(c);a.push(c.P);F(c.P)&2&&Hd(a,8)}
function me(a,b,c){Na(c);return G(a,b,c)}
function ne(){var a=new oe;return G(a,1,1)}
function pe(a,b){return null==a?b:a}
function qe(a,b){return pe(Zd(a,b),"")}
;var re;function se(a){switch(typeof a){case "number":return isFinite(a)?a:String(a);case "object":if(a)if(Array.isArray(a)){if(0!==(F(a)&128))return a=Array.prototype.slice.call(a),Td(a),a}else{if(yd(a))return ud(a);if(a instanceof Cd){var b=a.h;return null==b?"":"string"===typeof b?b:a.h=ud(b)}}}return a}
;function te(a,b,c,d,e){if(null!=a){if(Array.isArray(a))a=e&&0==a.length&&F(a)&1?void 0:ue(a,b,c,void 0!==d,e);else if(Md(a)){var f={},g;for(g in a)f[g]=te(a[g],b,c,d,e);a=f}else a=b(a,d);return a}}
function ue(a,b,c,d,e){var f=F(a);d=d?!!(f&16):void 0;a=Array.prototype.slice.call(a);for(var g=0;g<a.length;g++)a[g]=te(a[g],b,c,d,e);c(f,a);return a}
function ve(a){return a.Jc===Ld?a.toJSON():se(a)}
function we(a,b){a&128&&Td(b)}
;function xe(a,b,c){c=void 0===c?Kd:c;if(null!=a){if(sd&&a instanceof Uint8Array)return b?a:new Uint8Array(a);if(Array.isArray(a)){var d=F(a);if(d&2)return a;if(b&&!(d&32)&&(d&16||0===d))return Gd(a,d|18),a;a=ue(a,xe,d&4?Kd:c,!0,!1);b=F(a);b&4&&b&2&&Object.freeze(a);return a}return a.Jc===Ld?ye(a):a}}
function ze(a,b,c,d,e,f,g){(a=a.i&&a.i[c])?(d=0<a.length?a[0].constructor:void 0,f=F(a),f&2||(a=hb(a,ye),Kd(f,a),Object.freeze(a)),ke(b,d,c,a,e)):G(b,c,xe(d,f,g),e)}
function ye(a){if(F(a.P)&2)return a;a=Ae(a,!0);Ed(a.P,18);return a}
function Ae(a,b){var c=a.P,d=[];Ed(d,16);var e=a.constructor.h;e&&d.push(e);e=a.xa;if(e){d.length=c.length;var f={};d[d.length-1]=f}0!==(F(c)&128)&&Td(d);b=b||F(a.P)&2?Kd:Jd;f=a.constructor;F(d);re=d;d=new f(d);re=void 0;a.wd&&(d.wd=a.wd.slice());f=!!(F(c)&16);for(var g=e?c.length-1:c.length,h=0;h<g;h++)ze(a,d,h-a.fb,c[h],!1,f,b);if(e)for(var l in e)c=e[l],g=+l,Number.isNaN(g),ze(a,d,g,c,!0,f,b);return d}
function he(a){if(!(F(a.P)&2))return a;var b=Ae(a,!1);b.m=a;return b}
;function I(a,b,c,d){null==a&&(a=re);re=void 0;var e=this.constructor.h;if(null==a){a=e?[e]:[];var f=48;var g=!0;d&&(f|=128);Gd(a,f)}else{if(!Array.isArray(a))throw Error();if(e&&e!==a[0])throw Error();f=Ed(a,0)|32;g=0!==(16&f);if(d){if(f|=128,0<a.length){var h=a[a.length-1];if(Md(h)&&"g"in h){delete h.g;var l=!0,m;for(m in h){l=!1;break}l&&a.pop()}}}else if(128&f)throw Error();Gd(a,f)}this.fb=e?0:-1;this.i=void 0;this.P=a;a:{f=this.P.length;e=f-1;if(f&&(f=this.P[e],Md(f))){this.xa=f;this.j=e-this.fb;
break a}void 0!==b&&-1<b?(this.j=Math.max(b,e+1-this.fb),this.xa=void 0):this.j=Number.MAX_VALUE}if(!d&&this.xa&&"g"in this.xa)throw Error('Unexpected "g" flag in sparse object of message that is not a group type.');if(c){b=g&&!0;d=this.j;var n;for(g=0;g<c.length;g++)e=c[g],e<d?(e+=this.fb,(f=a[e])?Be(f,b):a[e]=Rd):(n||(n=Yd(this)),(f=n[e])?Be(f,b):n[e]=Rd)}F(this.P)}
I.prototype.toJSON=function(){var a=this.P,b;Qd?b=a:b=ue(a,ve,we,void 0,!1);return b};
function Ce(a){Qd=!0;try{return JSON.stringify(a.toJSON(),De)}finally{Qd=!1}}
I.prototype.clone=function(){return Ae(this,!1)};
function Be(a,b){if(Array.isArray(a)){var c=F(a),d=1;!b||c&2||(d|=16);(c&d)!==d&&Gd(a,c|d)}}
I.prototype.Jc=Ld;I.prototype.toString=function(){return this.P.toString()};
function De(a,b){return se(b)}
;function Ee(a){this.gd=a}
;function Fe(a,b,c){this.i=a;this.l=b;this.h=c||[];this.sb=new Map}
k=Fe.prototype;k.be=function(a){var b=Ja.apply(1,arguments),c=this.yc(b);c?c.push(new Ee(a)):this.Kd(a,b)};
k.Kd=function(a){this.sb.set(this.pd(Ja.apply(1,arguments)),[new Ee(a)])};
k.yc=function(){var a=this.pd(Ja.apply(0,arguments));return this.sb.has(a)?this.sb.get(a):void 0};
k.se=function(){var a=this.yc(Ja.apply(0,arguments));return a&&a.length?a[0]:void 0};
k.clear=function(){this.sb.clear()};
k.pd=function(){var a=Ja.apply(0,arguments);return a?a.join(","):"key"};function Ge(a,b){Fe.call(this,a,3,b)}
v(Ge,Fe);Ge.prototype.j=function(a){var b=Ja.apply(1,arguments),c=0,d=this.se(b);d&&(c=d.gd);this.Kd(c+a,b)};function He(a,b){Fe.call(this,a,2,b)}
v(He,Fe);He.prototype.j=function(a){this.be(a,Ja.apply(1,arguments))};function Ie(a){a&&"function"==typeof a.dispose&&a.dispose()}
;function Je(a){for(var b=0,c=arguments.length;b<c;++b){var d=arguments[b];Oa(d)?Je.apply(null,d):Ie(d)}}
;function K(){this.ra=this.ra;this.X=this.X}
K.prototype.ra=!1;K.prototype.h=function(){return this.ra};
K.prototype.dispose=function(){this.ra||(this.ra=!0,this.L())};
function Ke(a,b){Le(a,Xa(Ie,b))}
function Le(a,b){a.ra?b():(a.X||(a.X=[]),a.X.push(b))}
K.prototype.L=function(){if(this.X)for(;this.X.length;)this.X.shift()()};function Me(a,b){this.type=a;this.h=this.target=b;this.defaultPrevented=this.j=!1}
Me.prototype.stopPropagation=function(){this.j=!0};
Me.prototype.preventDefault=function(){this.defaultPrevented=!0};function Ne(a){var b=B("window.location.href");null==a&&(a='Unknown Error of type "null/undefined"');if("string"===typeof a)return{message:a,name:"Unknown error",lineNumber:"Not available",fileName:b,stack:"Not available"};var c=!1;try{var d=a.lineNumber||a.line||"Not available"}catch(g){d="Not available",c=!0}try{var e=a.fileName||a.filename||a.sourceURL||y.$googDebugFname||b}catch(g){e="Not available",c=!0}b=Oe(a);if(!(!c&&a.lineNumber&&a.fileName&&a.stack&&a.message&&a.name)){c=a.message;if(null==
c){if(a.constructor&&a.constructor instanceof Function){if(a.constructor.name)c=a.constructor.name;else if(c=a.constructor,Pe[c])c=Pe[c];else{c=String(c);if(!Pe[c]){var f=/function\s+([^\(]+)/m.exec(c);Pe[c]=f?f[1]:"[Anonymous]"}c=Pe[c]}c='Unknown Error of type "'+c+'"'}else c="Unknown Error of unknown type";"function"===typeof a.toString&&Object.prototype.toString!==a.toString&&(c+=": "+a.toString())}return{message:c,name:a.name||"UnknownError",lineNumber:d,fileName:e,stack:b||"Not available"}}a.stack=
b;return{message:a.message,name:a.name,lineNumber:a.lineNumber,fileName:a.fileName,stack:a.stack}}
function Oe(a,b){b||(b={});b[Qe(a)]=!0;var c=a.stack||"";(a=a.cause)&&!b[Qe(a)]&&(c+="\nCaused by: ",a.stack&&0==a.stack.indexOf(a.toString())||(c+="string"===typeof a?a:a.message+"\n"),c+=Oe(a,b));return c}
function Qe(a){var b="";"function"===typeof a.toString&&(b=""+a);return b+a.stack}
var Pe={};var Re=function(){if(!y.addEventListener||!Object.defineProperty)return!1;var a=!1,b=Object.defineProperty({},"passive",{get:function(){a=!0}});
try{y.addEventListener("test",function(){},b),y.removeEventListener("test",function(){},b)}catch(c){}return a}();function Se(a,b){Me.call(this,a?a.type:"");this.relatedTarget=this.h=this.target=null;this.button=this.screenY=this.screenX=this.clientY=this.clientX=0;this.key="";this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.altKey=this.ctrlKey=!1;this.state=null;this.pointerId=0;this.pointerType="";this.i=null;a&&this.init(a,b)}
Ya(Se,Me);var Te={2:"touch",3:"pen",4:"mouse"};
Se.prototype.init=function(a,b){var c=this.type=a.type,d=a.changedTouches&&a.changedTouches.length?a.changedTouches[0]:null;this.target=a.target||a.srcElement;this.h=b;if(b=a.relatedTarget){if(Yc){a:{try{Uc(b.nodeName);var e=!0;break a}catch(f){}e=!1}e||(b=null)}}else"mouseover"==c?b=a.fromElement:"mouseout"==c&&(b=a.toElement);this.relatedTarget=b;d?(this.clientX=void 0!==d.clientX?d.clientX:d.pageX,this.clientY=void 0!==d.clientY?d.clientY:d.pageY,this.screenX=d.screenX||0,this.screenY=d.screenY||
0):(this.clientX=void 0!==a.clientX?a.clientX:a.pageX,this.clientY=void 0!==a.clientY?a.clientY:a.pageY,this.screenX=a.screenX||0,this.screenY=a.screenY||0);this.button=a.button;this.keyCode=a.keyCode||0;this.key=a.key||"";this.charCode=a.charCode||("keypress"==c?a.keyCode:0);this.ctrlKey=a.ctrlKey;this.altKey=a.altKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.pointerId=a.pointerId||0;this.pointerType="string"===typeof a.pointerType?a.pointerType:Te[a.pointerType]||"";this.state=a.state;
this.i=a;a.defaultPrevented&&Se.va.preventDefault.call(this)};
Se.prototype.stopPropagation=function(){Se.va.stopPropagation.call(this);this.i.stopPropagation?this.i.stopPropagation():this.i.cancelBubble=!0};
Se.prototype.preventDefault=function(){Se.va.preventDefault.call(this);var a=this.i;a.preventDefault?a.preventDefault():a.returnValue=!1};var Ue="closure_listenable_"+(1E6*Math.random()|0);var Ve=0;function We(a,b,c,d,e){this.listener=a;this.proxy=null;this.src=b;this.type=c;this.capture=!!d;this.hc=e;this.key=++Ve;this.Pb=this.ac=!1}
function Xe(a){a.Pb=!0;a.listener=null;a.proxy=null;a.src=null;a.hc=null}
;function Ye(a){this.src=a;this.listeners={};this.h=0}
Ye.prototype.add=function(a,b,c,d,e){var f=a.toString();a=this.listeners[f];a||(a=this.listeners[f]=[],this.h++);var g=Ze(a,b,d,e);-1<g?(b=a[g],c||(b.ac=!1)):(b=new We(b,this.src,f,!!d,e),b.ac=c,a.push(b));return b};
Ye.prototype.remove=function(a,b,c,d){a=a.toString();if(!(a in this.listeners))return!1;var e=this.listeners[a];b=Ze(e,b,c,d);return-1<b?(Xe(e[b]),Array.prototype.splice.call(e,b,1),0==e.length&&(delete this.listeners[a],this.h--),!0):!1};
function $e(a,b){var c=b.type;c in a.listeners&&kb(a.listeners[c],b)&&(Xe(b),0==a.listeners[c].length&&(delete a.listeners[c],a.h--))}
function Ze(a,b,c,d){for(var e=0;e<a.length;++e){var f=a[e];if(!f.Pb&&f.listener==b&&f.capture==!!c&&f.hc==d)return e}return-1}
;var af="closure_lm_"+(1E6*Math.random()|0),bf={},cf=0;function df(a,b,c,d,e){if(d&&d.once)ef(a,b,c,d,e);else if(Array.isArray(b))for(var f=0;f<b.length;f++)df(a,b[f],c,d,e);else c=kf(c),a&&a[Ue]?a.La(b,c,Pa(d)?!!d.capture:!!d,e):lf(a,b,c,!1,d,e)}
function lf(a,b,c,d,e,f){if(!b)throw Error("Invalid event type");var g=Pa(e)?!!e.capture:!!e,h=mf(a);h||(a[af]=h=new Ye(a));c=h.add(b,c,d,g,f);if(!c.proxy){d=nf();c.proxy=d;d.src=a;d.listener=c;if(a.addEventListener)Re||(e=g),void 0===e&&(e=!1),a.addEventListener(b.toString(),d,e);else if(a.attachEvent)a.attachEvent(of(b.toString()),d);else if(a.addListener&&a.removeListener)a.addListener(d);else throw Error("addEventListener and attachEvent are unavailable.");cf++}}
function nf(){function a(c){return b.call(a.src,a.listener,c)}
var b=pf;return a}
function ef(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)ef(a,b[f],c,d,e);else c=kf(c),a&&a[Ue]?a.l.add(String(b),c,!0,Pa(d)?!!d.capture:!!d,e):lf(a,b,c,!0,d,e)}
function qf(a,b,c,d,e){if(Array.isArray(b))for(var f=0;f<b.length;f++)qf(a,b[f],c,d,e);else(d=Pa(d)?!!d.capture:!!d,c=kf(c),a&&a[Ue])?a.l.remove(String(b),c,d,e):a&&(a=mf(a))&&(b=a.listeners[b.toString()],a=-1,b&&(a=Ze(b,c,d,e)),(c=-1<a?b[a]:null)&&rf(c))}
function rf(a){if("number"!==typeof a&&a&&!a.Pb){var b=a.src;if(b&&b[Ue])$e(b.l,a);else{var c=a.type,d=a.proxy;b.removeEventListener?b.removeEventListener(c,d,a.capture):b.detachEvent?b.detachEvent(of(c),d):b.addListener&&b.removeListener&&b.removeListener(d);cf--;(c=mf(b))?($e(c,a),0==c.h&&(c.src=null,b[af]=null)):Xe(a)}}}
function of(a){return a in bf?bf[a]:bf[a]="on"+a}
function pf(a,b){if(a.Pb)a=!0;else{b=new Se(b,this);var c=a.listener,d=a.hc||a.src;a.ac&&rf(a);a=c.call(d,b)}return a}
function mf(a){a=a[af];return a instanceof Ye?a:null}
var sf="__closure_events_fn_"+(1E9*Math.random()>>>0);function kf(a){if("function"===typeof a)return a;a[sf]||(a[sf]=function(b){return a.handleEvent(b)});
return a[sf]}
;function tf(){K.call(this);this.l=new Ye(this);this.Yd=this;this.Oa=null}
Ya(tf,K);tf.prototype[Ue]=!0;tf.prototype.addEventListener=function(a,b,c,d){df(this,a,b,c,d)};
tf.prototype.removeEventListener=function(a,b,c,d){qf(this,a,b,c,d)};
function uf(a,b){var c=a.Oa;if(c){var d=[];for(var e=1;c;c=c.Oa)d.push(c),++e}a=a.Yd;c=b.type||b;"string"===typeof b?b=new Me(b,a):b instanceof Me?b.target=b.target||a:(e=b,b=new Me(c,a),wb(b,e));e=!0;if(d)for(var f=d.length-1;!b.j&&0<=f;f--){var g=b.h=d[f];e=vf(g,c,!0,b)&&e}b.j||(g=b.h=a,e=vf(g,c,!0,b)&&e,b.j||(e=vf(g,c,!1,b)&&e));if(d)for(f=0;!b.j&&f<d.length;f++)g=b.h=d[f],e=vf(g,c,!1,b)&&e}
tf.prototype.L=function(){tf.va.L.call(this);if(this.l){var a=this.l,b=0,c;for(c in a.listeners){for(var d=a.listeners[c],e=0;e<d.length;e++)++b,Xe(d[e]);delete a.listeners[c];a.h--}}this.Oa=null};
tf.prototype.La=function(a,b,c,d){return this.l.add(String(a),b,!1,c,d)};
function vf(a,b,c,d){b=a.l.listeners[String(b)];if(!b)return!0;b=b.concat();for(var e=!0,f=0;f<b.length;++f){var g=b[f];if(g&&!g.Pb&&g.capture==c){var h=g.listener,l=g.hc||g.src;g.ac&&$e(a.l,g);e=!1!==h.call(l,d)&&e}}return e&&!d.defaultPrevented}
;function wf(a,b){this.j=a;this.l=b;this.i=0;this.h=null}
wf.prototype.get=function(){if(0<this.i){this.i--;var a=this.h;this.h=a.next;a.next=null}else a=this.j();return a};
function xf(a,b){a.l(b);100>a.i&&(a.i++,b.next=a.h,a.h=b)}
;function yf(a,b){return a+Math.random()*(b-a)}
;function zf(a,b){this.x=void 0!==a?a:0;this.y=void 0!==b?b:0}
k=zf.prototype;k.clone=function(){return new zf(this.x,this.y)};
k.equals=function(a){return a instanceof zf&&(this==a?!0:this&&a?this.x==a.x&&this.y==a.y:!1)};
k.ceil=function(){this.x=Math.ceil(this.x);this.y=Math.ceil(this.y);return this};
k.floor=function(){this.x=Math.floor(this.x);this.y=Math.floor(this.y);return this};
k.round=function(){this.x=Math.round(this.x);this.y=Math.round(this.y);return this};
k.scale=function(a,b){this.x*=a;this.y*="number"===typeof b?b:a;return this};function Af(a,b){this.width=a;this.height=b}
k=Af.prototype;k.clone=function(){return new Af(this.width,this.height)};
k.aspectRatio=function(){return this.width/this.height};
k.ceil=function(){this.width=Math.ceil(this.width);this.height=Math.ceil(this.height);return this};
k.floor=function(){this.width=Math.floor(this.width);this.height=Math.floor(this.height);return this};
k.round=function(){this.width=Math.round(this.width);this.height=Math.round(this.height);return this};
k.scale=function(a,b){this.width*=a;this.height*="number"===typeof b?b:a;return this};function Bf(a){var b=document;return"string"===typeof a?b.getElementById(a):a}
function Cf(a){var b=document;a=String(a);"application/xhtml+xml"===b.contentType&&(a=a.toLowerCase());return b.createElement(a)}
function Df(a,b){for(var c=0;a;){if(b(a))return a;a=a.parentNode;c++}return null}
;var Ef;function Ff(){var a=y.MessageChannel;"undefined"===typeof a&&"undefined"!==typeof window&&window.postMessage&&window.addEventListener&&!D("Presto")&&(a=function(){var e=Cf("IFRAME");e.style.display="none";document.documentElement.appendChild(e);var f=e.contentWindow;e=f.document;e.open();e.close();var g="callImmediate"+Math.random(),h="file:"==f.location.protocol?"*":f.location.protocol+"//"+f.location.host;e=Wa(function(l){if(("*"==h||l.origin==h)&&l.data==g)this.port1.onmessage()},this);
f.addEventListener("message",e,!1);this.port1={};this.port2={postMessage:function(){f.postMessage(g,h)}}});
if("undefined"!==typeof a&&!ac()){var b=new a,c={},d=c;b.port1.onmessage=function(){if(void 0!==c.next){c=c.next;var e=c.fd;c.fd=null;e()}};
return function(e){d.next={fd:e};d=d.next;b.port2.postMessage(0)}}return function(e){y.setTimeout(e,0)}}
;function Gf(){this.i=this.h=null}
Gf.prototype.add=function(a,b){var c=Hf.get();c.set(a,b);this.i?this.i.next=c:this.h=c;this.i=c};
Gf.prototype.remove=function(){var a=null;this.h&&(a=this.h,this.h=this.h.next,this.h||(this.i=null),a.next=null);return a};
var Hf=new wf(function(){return new If},function(a){return a.reset()});
function If(){this.next=this.scope=this.h=null}
If.prototype.set=function(a,b){this.h=a;this.scope=b;this.next=null};
If.prototype.reset=function(){this.next=this.scope=this.h=null};var Jf,Kf=!1,Lf=new Gf;function Mf(a,b){Jf||Nf();Kf||(Jf(),Kf=!0);Lf.add(a,b)}
function Nf(){if(y.Promise&&y.Promise.resolve){var a=y.Promise.resolve(void 0);Jf=function(){a.then(Of)}}else Jf=function(){var b=Of;
"function"!==typeof y.setImmediate||y.Window&&y.Window.prototype&&!bc()&&y.Window.prototype.setImmediate==y.setImmediate?(Ef||(Ef=Ff()),Ef(b)):y.setImmediate(b)}}
function Of(){for(var a;a=Lf.remove();){try{a.h.call(a.scope)}catch(b){Jc(b)}xf(Hf,a)}Kf=!1}
;function Pf(a){this.h=0;this.v=void 0;this.l=this.i=this.j=null;this.m=this.s=!1;if(a!=cb)try{var b=this;a.call(void 0,function(c){Qf(b,2,c)},function(c){Qf(b,3,c)})}catch(c){Qf(this,3,c)}}
function Rf(){this.next=this.context=this.i=this.j=this.h=null;this.l=!1}
Rf.prototype.reset=function(){this.context=this.i=this.j=this.h=null;this.l=!1};
var Sf=new wf(function(){return new Rf},function(a){a.reset()});
function Tf(a,b,c){var d=Sf.get();d.j=a;d.i=b;d.context=c;return d}
function Uf(a){return new Pf(function(b,c){c(a)})}
Pf.prototype.then=function(a,b,c){return Vf(this,"function"===typeof a?a:null,"function"===typeof b?b:null,c)};
Pf.prototype.$goog_Thenable=!0;k=Pf.prototype;k.sc=function(a,b){return Vf(this,null,a,b)};
k.catch=Pf.prototype.sc;k.cancel=function(a){if(0==this.h){var b=new Wf(a);Mf(function(){Xf(this,b)},this)}};
function Xf(a,b){if(0==a.h)if(a.j){var c=a.j;if(c.i){for(var d=0,e=null,f=null,g=c.i;g&&(g.l||(d++,g.h==a&&(e=g),!(e&&1<d)));g=g.next)e||(f=g);e&&(0==c.h&&1==d?Xf(c,b):(f?(d=f,d.next==c.l&&(c.l=d),d.next=d.next.next):Yf(c),Zf(c,e,3,b)))}a.j=null}else Qf(a,3,b)}
function $f(a,b){a.i||2!=a.h&&3!=a.h||ag(a);a.l?a.l.next=b:a.i=b;a.l=b}
function Vf(a,b,c,d){var e=Tf(null,null,null);e.h=new Pf(function(f,g){e.j=b?function(h){try{var l=b.call(d,h);f(l)}catch(m){g(m)}}:f;
e.i=c?function(h){try{var l=c.call(d,h);void 0===l&&h instanceof Wf?g(h):f(l)}catch(m){g(m)}}:g});
e.h.j=a;$f(a,e);return e.h}
k.lf=function(a){this.h=0;Qf(this,2,a)};
k.mf=function(a){this.h=0;Qf(this,3,a)};
function Qf(a,b,c){if(0==a.h){a===c&&(b=3,c=new TypeError("Promise cannot resolve to itself"));a.h=1;a:{var d=c,e=a.lf,f=a.mf;if(d instanceof Pf){$f(d,Tf(e||cb,f||null,a));var g=!0}else{if(d)try{var h=!!d.$goog_Thenable}catch(m){h=!1}else h=!1;if(h)d.then(e,f,a),g=!0;else{if(Pa(d))try{var l=d.then;if("function"===typeof l){bg(d,l,e,f,a);g=!0;break a}}catch(m){f.call(a,m);g=!0;break a}g=!1}}}g||(a.v=c,a.h=b,a.j=null,ag(a),3!=b||c instanceof Wf||cg(a,c))}}
function bg(a,b,c,d,e){function f(l){h||(h=!0,d.call(e,l))}
function g(l){h||(h=!0,c.call(e,l))}
var h=!1;try{b.call(a,g,f)}catch(l){f(l)}}
function ag(a){a.s||(a.s=!0,Mf(a.ne,a))}
function Yf(a){var b=null;a.i&&(b=a.i,a.i=b.next,b.next=null);a.i||(a.l=null);return b}
k.ne=function(){for(var a;a=Yf(this);)Zf(this,a,this.h,this.v);this.s=!1};
function Zf(a,b,c,d){if(3==c&&b.i&&!b.l)for(;a&&a.m;a=a.j)a.m=!1;if(b.h)b.h.j=null,dg(b,c,d);else try{b.l?b.j.call(b.context):dg(b,c,d)}catch(e){eg.call(null,e)}xf(Sf,b)}
function dg(a,b,c){2==b?a.j.call(a.context,c):a.i&&a.i.call(a.context,c)}
function cg(a,b){a.m=!0;Mf(function(){a.m&&eg.call(null,b)})}
var eg=Jc;function Wf(a){$a.call(this,a)}
Ya(Wf,$a);Wf.prototype.name="cancel";function fg(a,b){tf.call(this);this.j=a||1;this.i=b||y;this.m=Wa(this.jf,this);this.s=Date.now()}
Ya(fg,tf);k=fg.prototype;k.enabled=!1;k.za=null;function gg(a,b){a.j=b;a.za&&a.enabled?(a.stop(),a.start()):a.za&&a.stop()}
k.jf=function(){if(this.enabled){var a=Date.now()-this.s;0<a&&a<.8*this.j?this.za=this.i.setTimeout(this.m,this.j-a):(this.za&&(this.i.clearTimeout(this.za),this.za=null),uf(this,"tick"),this.enabled&&(this.stop(),this.start()))}};
k.start=function(){this.enabled=!0;this.za||(this.za=this.i.setTimeout(this.m,this.j),this.s=Date.now())};
k.stop=function(){this.enabled=!1;this.za&&(this.i.clearTimeout(this.za),this.za=null)};
k.L=function(){fg.va.L.call(this);this.stop();delete this.i};
function hg(a,b,c){if("function"===typeof a)c&&(a=Wa(a,c));else if(a&&"function"==typeof a.handleEvent)a=Wa(a.handleEvent,a);else throw Error("Invalid listener argument");return 2147483647<Number(b)?-1:y.setTimeout(a,b||0)}
;function ig(a){K.call(this);this.H=a;this.j=new Map;this.v=new Set;this.l=0;this.m=100;this.flushInterval=3E4;this.i=new fg(this.flushInterval);this.i.La("tick",this.pb,!1,this);Ke(this,this.i);this.s=!1}
v(ig,K);k=ig.prototype;k.sendIsolatedPayload=function(a){this.s=a;this.m=1};
function jg(a){a.i.enabled||a.i.start();a.l++;a.l>=a.m&&a.pb()}
k.pb=function(){var a=this.j.values();a=[].concat(ha(a)).filter(function(b){return b.sb.size});
a.length&&this.H.flush(a,this.s);kg(a);this.l=0;this.i.enabled&&this.i.stop()};
k.bd=function(a){var b=Ja.apply(1,arguments);this.j.has(a)||this.j.set(a,new Ge(a,b))};
k.cd=function(a){var b=Ja.apply(1,arguments);this.j.has(a)||this.j.set(a,new He(a,b))};
function lg(a,b){return a.v.has(b)?void 0:a.j.get(b)}
k.uc=function(a){this.Wd.apply(this,[a,1].concat(ha(Ja.apply(1,arguments))))};
k.Wd=function(a,b){var c=Ja.apply(2,arguments),d=lg(this,a);d&&d instanceof Ge&&(d.j(b,c),jg(this))};
k.vc=function(a,b){var c=Ja.apply(2,arguments),d=lg(this,a);d&&d instanceof He&&(d.j(b,c),jg(this))};
function kg(a){for(var b=0;b<a.length;b++)a[b].clear()}
;function mg(a){this.h=a;this.h.bd("/client_streamz/bg/fiec",{Lb:3,Kb:"rk"},{Lb:2,Kb:"ec"})}
function ng(a,b,c){a.h.uc("/client_streamz/bg/fiec",b,c)}
function og(a){this.h=a;this.h.cd("/client_streamz/bg/fil",{Lb:3,Kb:"rk"})}
function pg(a){this.h=a;this.h.bd("/client_streamz/bg/fsc",{Lb:3,Kb:"rk"})}
function qg(a){this.h=a;this.h.cd("/client_streamz/bg/fsl",{Lb:3,Kb:"rk"})}
;var rg={toString:function(a){var b=[],c=0;a-=-2147483648;b[c++]="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ".charAt(a%52);for(a=Math.floor(a/52);0<a;)b[c++]="abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789".charAt(a%62),a=Math.floor(a/62);return b.join("")}};function sg(a){function b(){c-=d;c-=e;c^=e>>>13;d-=e;d-=c;d^=c<<8;e-=c;e-=d;e^=d>>>13;c-=d;c-=e;c^=e>>>12;d-=e;d-=c;d^=c<<16;e-=c;e-=d;e^=d>>>5;c-=d;c-=e;c^=e>>>3;d-=e;d-=c;d^=c<<10;e-=c;e-=d;e^=d>>>15}
a=tg(a);for(var c=2654435769,d=2654435769,e=314159265,f=a.length,g=f,h=0;12<=g;g-=12,h+=12)c+=ug(a,h),d+=ug(a,h+4),e+=ug(a,h+8),b();e+=f;switch(g){case 11:e+=a[h+10]<<24;case 10:e+=a[h+9]<<16;case 9:e+=a[h+8]<<8;case 8:d+=a[h+7]<<24;case 7:d+=a[h+6]<<16;case 6:d+=a[h+5]<<8;case 5:d+=a[h+4];case 4:c+=a[h+3]<<24;case 3:c+=a[h+2]<<16;case 2:c+=a[h+1]<<8;case 1:c+=a[h+0]}b();return rg.toString(e)}
function tg(a){for(var b=[],c=0;c<a.length;c++)b.push(a.charCodeAt(c));return b}
function ug(a,b){return a[b+0]+(a[b+1]<<8)+(a[b+2]<<16)+(a[b+3]<<24)}
;function vg(a){I.call(this,a)}
v(vg,I);var wg=[1,2,3];function xg(a){I.call(this,a)}
v(xg,I);var yg=[1,2,3];function zg(a){I.call(this,a,-1,Ag)}
v(zg,I);var Ag=[1];function Bg(a){I.call(this,a,-1,Cg)}
v(Bg,I);var Cg=[3,6,4];function Dg(a){I.call(this,a,-1,Eg)}
v(Dg,I);var Eg=[1];function Fg(a){if(!a)return"";if(/^about:(?:blank|srcdoc)$/.test(a))return window.origin||"";a=a.split("#")[0].split("?")[0];a=a.toLowerCase();0==a.indexOf("//")&&(a=window.location.protocol+a);/^[\w\-]*:\/\//.test(a)||(a=window.location.href);var b=a.substring(a.indexOf("://")+3),c=b.indexOf("/");-1!=c&&(b=b.substring(0,c));c=a.substring(0,a.indexOf("://"));if(!c)throw Error("URI is missing protocol: "+a);if("http"!==c&&"https"!==c&&"chrome-extension"!==c&&"moz-extension"!==c&&"file"!==c&&"android-app"!==
c&&"chrome-search"!==c&&"chrome-untrusted"!==c&&"chrome"!==c&&"app"!==c&&"devtools"!==c)throw Error("Invalid URI scheme in origin: "+c);a="";var d=b.indexOf(":");if(-1!=d){var e=b.substring(d+1);b=b.substring(0,d);if("http"===c&&"80"!==e||"https"===c&&"443"!==e)a=":"+e}return c+"://"+b+a}
;function Gg(){function a(){e[0]=1732584193;e[1]=4023233417;e[2]=2562383102;e[3]=271733878;e[4]=3285377520;n=m=0}
function b(q){for(var u=g,t=0;64>t;t+=4)u[t/4]=q[t]<<24|q[t+1]<<16|q[t+2]<<8|q[t+3];for(t=16;80>t;t++)q=u[t-3]^u[t-8]^u[t-14]^u[t-16],u[t]=(q<<1|q>>>31)&4294967295;q=e[0];var A=e[1],C=e[2],E=e[3],M=e[4];for(t=0;80>t;t++){if(40>t)if(20>t){var Q=E^A&(C^E);var S=1518500249}else Q=A^C^E,S=1859775393;else 60>t?(Q=A&C|E&(A|C),S=2400959708):(Q=A^C^E,S=3395469782);Q=((q<<5|q>>>27)&4294967295)+Q+M+S+u[t]&4294967295;M=E;E=C;C=(A<<30|A>>>2)&4294967295;A=q;q=Q}e[0]=e[0]+q&4294967295;e[1]=e[1]+A&4294967295;e[2]=
e[2]+C&4294967295;e[3]=e[3]+E&4294967295;e[4]=e[4]+M&4294967295}
function c(q,u){if("string"===typeof q){q=unescape(encodeURIComponent(q));for(var t=[],A=0,C=q.length;A<C;++A)t.push(q.charCodeAt(A));q=t}u||(u=q.length);t=0;if(0==m)for(;t+64<u;)b(q.slice(t,t+64)),t+=64,n+=64;for(;t<u;)if(f[m++]=q[t++],n++,64==m)for(m=0,b(f);t+64<u;)b(q.slice(t,t+64)),t+=64,n+=64}
function d(){var q=[],u=8*n;56>m?c(h,56-m):c(h,64-(m-56));for(var t=63;56<=t;t--)f[t]=u&255,u>>>=8;b(f);for(t=u=0;5>t;t++)for(var A=24;0<=A;A-=8)q[u++]=e[t]>>A&255;return q}
for(var e=[],f=[],g=[],h=[128],l=1;64>l;++l)h[l]=0;var m,n;a();return{reset:a,update:c,digest:d,je:function(){for(var q=d(),u="",t=0;t<q.length;t++)u+="0123456789ABCDEF".charAt(Math.floor(q[t]/16))+"0123456789ABCDEF".charAt(q[t]%16);return u}}}
;function Hg(a,b,c){var d=String(y.location.href);return d&&a&&b?[b,Ig(Fg(d),a,c||null)].join(" "):null}
function Ig(a,b,c){var d=[],e=[];if(1==(Array.isArray(c)?2:1))return e=[b,a],fb(d,function(h){e.push(h)}),Jg(e.join(" "));
var f=[],g=[];fb(c,function(h){g.push(h.key);f.push(h.value)});
c=Math.floor((new Date).getTime()/1E3);e=0==f.length?[c,b,a]:[f.join(":"),c,b,a];fb(d,function(h){e.push(h)});
a=Jg(e.join(" "));a=[c,a];0==g.length||a.push(g.join(""));return a.join("_")}
function Jg(a){var b=Gg();b.update(a);return b.je().toLowerCase()}
;var Kg={};function Lg(a){this.h=a||{cookie:""}}
k=Lg.prototype;k.isEnabled=function(){if(!y.navigator.cookieEnabled)return!1;if(this.h.cookie)return!0;this.set("TESTCOOKIESENABLED","1",{kc:60});if("1"!==this.get("TESTCOOKIESENABLED"))return!1;this.remove("TESTCOOKIESENABLED");return!0};
k.set=function(a,b,c){var d=!1;if("object"===typeof c){var e=c.Hg;d=c.secure||!1;var f=c.domain||void 0;var g=c.path||void 0;var h=c.kc}if(/[;=\s]/.test(a))throw Error('Invalid cookie name "'+a+'"');if(/[;\r\n]/.test(b))throw Error('Invalid cookie value "'+b+'"');void 0===h&&(h=-1);c=f?";domain="+f:"";g=g?";path="+g:"";d=d?";secure":"";h=0>h?"":0==h?";expires="+(new Date(1970,1,1)).toUTCString():";expires="+(new Date(Date.now()+1E3*h)).toUTCString();this.h.cookie=a+"="+b+c+g+h+d+(null!=e?";samesite="+
e:"")};
k.get=function(a,b){for(var c=a+"=",d=(this.h.cookie||"").split(";"),e=0,f;e<d.length;e++){f=Ib(d[e]);if(0==f.lastIndexOf(c,0))return f.slice(c.length);if(f==a)return""}return b};
k.remove=function(a,b,c){var d=void 0!==this.get(a);this.set(a,"",{kc:0,path:b,domain:c});return d};
k.Bc=function(){return Mg(this).keys};
k.clear=function(){for(var a=Mg(this).keys,b=a.length-1;0<=b;b--)this.remove(a[b])};
function Mg(a){a=(a.h.cookie||"").split(";");for(var b=[],c=[],d,e,f=0;f<a.length;f++)e=Ib(a[f]),d=e.indexOf("="),-1==d?(b.push(""),c.push(e)):(b.push(e.substring(0,d)),c.push(e.substring(d+1)));return{keys:b,values:c}}
var Ng=new Lg("undefined"==typeof document?null:document);function Og(a){return!!Kg.FPA_SAMESITE_PHASE2_MOD||!(void 0===a||!a)}
function Pg(a){a=void 0===a?!1:a;var b=y.__SAPISID||y.__APISID||y.__3PSAPISID||y.__OVERRIDE_SID;Og(a)&&(b=b||y.__1PSAPISID);if(b)return!0;var c=new Lg(document);b=c.get("SAPISID")||c.get("APISID")||c.get("__Secure-3PAPISID")||c.get("SID")||c.get("OSID");Og(a)&&(b=b||c.get("__Secure-1PAPISID"));return!!b}
function Qg(a,b,c,d){(a=y[a])||(a=(new Lg(document)).get(b));return a?Hg(a,c,d):null}
function Rg(a,b){b=void 0===b?!1:b;var c=Fg(String(y.location.href)),d=[];if(Pg(b)){c=0==c.indexOf("https:")||0==c.indexOf("chrome-extension:")||0==c.indexOf("moz-extension:");var e=c?y.__SAPISID:y.__APISID;e||(e=new Lg(document),e=e.get(c?"SAPISID":"APISID")||e.get("__Secure-3PAPISID"));(e=e?Hg(e,c?"SAPISIDHASH":"APISIDHASH",a):null)&&d.push(e);c&&Og(b)&&((b=Qg("__1PSAPISID","__Secure-1PAPISID","SAPISID1PHASH",a))&&d.push(b),(a=Qg("__3PSAPISID","__Secure-3PAPISID","SAPISID3PHASH",a))&&d.push(a))}return 0==
d.length?null:d.join(" ")}
;function Sg(a){I.call(this,a,-1,Tg)}
v(Sg,I);var Tg=[2];function Ug(a){this.h=this.i=this.j=a}
Ug.prototype.reset=function(){this.h=this.i=this.j};
Ug.prototype.getValue=function(){return this.i};function Vg(a){var b=[];Wg(new Xg,a,b);return b.join("")}
function Xg(){}
function Wg(a,b,c){if(null==b)c.push("null");else{if("object"==typeof b){if(Array.isArray(b)){var d=b;b=d.length;c.push("[");for(var e="",f=0;f<b;f++)c.push(e),Wg(a,d[f],c),e=",";c.push("]");return}if(b instanceof String||b instanceof Number||b instanceof Boolean)b=b.valueOf();else{c.push("{");e="";for(d in b)Object.prototype.hasOwnProperty.call(b,d)&&(f=b[d],"function"!=typeof f&&(c.push(e),Yg(d,c),c.push(":"),Wg(a,f,c),e=","));c.push("}");return}}switch(typeof b){case "string":Yg(b,c);break;case "number":c.push(isFinite(b)&&
!isNaN(b)?String(b):"null");break;case "boolean":c.push(String(b));break;case "function":c.push("null");break;default:throw Error("Unknown type: "+typeof b);}}}
var Zg={'"':'\\"',"\\":"\\\\","/":"\\/","\b":"\\b","\f":"\\f","\n":"\\n","\r":"\\r","\t":"\\t","\v":"\\u000b"},$g=/\uffff/.test("\uffff")?/[\\"\x00-\x1f\x7f-\uffff]/g:/[\\"\x00-\x1f\x7f-\xff]/g;function Yg(a,b){b.push('"',a.replace($g,function(c){var d=Zg[c];d||(d="\\u"+(c.charCodeAt(0)|65536).toString(16).slice(1),Zg[c]=d);return d}),'"')}
;function ah(){}
ah.prototype.h=null;ah.prototype.getOptions=function(){var a;(a=this.h)||(a={},bh(this)&&(a[0]=!0,a[1]=!0),a=this.h=a);return a};var ch;function dh(){}
Ya(dh,ah);function eh(a){return(a=bh(a))?new ActiveXObject(a):new XMLHttpRequest}
function bh(a){if(!a.i&&"undefined"==typeof XMLHttpRequest&&"undefined"!=typeof ActiveXObject){for(var b=["MSXML2.XMLHTTP.6.0","MSXML2.XMLHTTP.3.0","MSXML2.XMLHTTP","Microsoft.XMLHTTP"],c=0;c<b.length;c++){var d=b[c];try{return new ActiveXObject(d),a.i=d}catch(e){}}throw Error("Could not create ActiveXObject. ActiveX might be disabled, or MSXML might not be installed");}return a.i}
ch=new dh;function fh(a){tf.call(this);this.headers=new Map;this.T=a||null;this.i=!1;this.M=this.F=null;this.m=this.aa="";this.j=this.Y=this.v=this.V=!1;this.s=0;this.H=null;this.ma="";this.ha=this.ia=!1}
Ya(fh,tf);var gh=/^https?$/i,hh=["POST","PUT"],ih=[];function jh(a,b,c,d,e,f,g){var h=new fh;ih.push(h);b&&h.La("complete",b);h.l.add("ready",h.he,!0,void 0,void 0);f&&(h.s=Math.max(0,f));g&&(h.ia=g);h.send(a,c,d,e)}
k=fh.prototype;k.he=function(){this.dispose();kb(ih,this)};
k.send=function(a,b,c,d){if(this.F)throw Error("[goog.net.XhrIo] Object is active with another request="+this.aa+"; newUri="+a);b=b?b.toUpperCase():"GET";this.aa=a;this.m="";this.V=!1;this.i=!0;this.F=this.T?eh(this.T):eh(ch);this.M=this.T?this.T.getOptions():ch.getOptions();this.F.onreadystatechange=Wa(this.Bd,this);try{this.getStatus(),this.Y=!0,this.F.open(b,String(a),!0),this.Y=!1}catch(g){this.getStatus();kh(this,g);return}a=c||"";c=new Map(this.headers);if(d)if(Object.getPrototypeOf(d)===Object.prototype)for(var e in d)c.set(e,
d[e]);else if("function"===typeof d.keys&&"function"===typeof d.get){e=r(d.keys());for(var f=e.next();!f.done;f=e.next())f=f.value,c.set(f,d.get(f))}else throw Error("Unknown input type for opt_headers: "+String(d));d=Array.from(c.keys()).find(function(g){return"content-type"==g.toLowerCase()});
e=y.FormData&&a instanceof y.FormData;!(0<=eb(hh,b))||d||e||c.set("Content-Type","application/x-www-form-urlencoded;charset=utf-8");b=r(c);for(d=b.next();!d.done;d=b.next())c=r(d.value),d=c.next().value,c=c.next().value,this.F.setRequestHeader(d,c);this.ma&&(this.F.responseType=this.ma);"withCredentials"in this.F&&this.F.withCredentials!==this.ia&&(this.F.withCredentials=this.ia);try{lh(this),0<this.s&&(this.ha=mh(this.F),this.getStatus(),this.ha?(this.F.timeout=this.s,this.F.ontimeout=Wa(this.Pd,
this)):this.H=hg(this.Pd,this.s,this)),this.getStatus(),this.v=!0,this.F.send(a),this.v=!1}catch(g){this.getStatus(),kh(this,g)}};
function mh(a){return Wc&&"number"===typeof a.timeout&&void 0!==a.ontimeout}
k.Pd=function(){"undefined"!=typeof Ma&&this.F&&(this.m="Timed out after "+this.s+"ms, aborting",this.getStatus(),uf(this,"timeout"),this.abort(8))};
function kh(a,b){a.i=!1;a.F&&(a.j=!0,a.F.abort(),a.j=!1);a.m=b;nh(a);oh(a)}
function nh(a){a.V||(a.V=!0,uf(a,"complete"),uf(a,"error"))}
k.abort=function(){this.F&&this.i&&(this.getStatus(),this.i=!1,this.j=!0,this.F.abort(),this.j=!1,uf(this,"complete"),uf(this,"abort"),oh(this))};
k.L=function(){this.F&&(this.i&&(this.i=!1,this.j=!0,this.F.abort(),this.j=!1),oh(this,!0));fh.va.L.call(this)};
k.Bd=function(){this.h()||(this.Y||this.v||this.j?ph(this):this.Ke())};
k.Ke=function(){ph(this)};
function ph(a){if(a.i&&"undefined"!=typeof Ma)if(a.M[1]&&4==qh(a)&&2==a.getStatus())a.getStatus();else if(a.v&&4==qh(a))hg(a.Bd,0,a);else if(uf(a,"readystatechange"),a.isComplete()){a.getStatus();a.i=!1;try{if(rh(a))uf(a,"complete"),uf(a,"success");else{try{var b=2<qh(a)?a.F.statusText:""}catch(c){b=""}a.m=b+" ["+a.getStatus()+"]";nh(a)}}finally{oh(a)}}}
function oh(a,b){if(a.F){lh(a);var c=a.F,d=a.M[0]?function(){}:null;
a.F=null;a.M=null;b||uf(a,"ready");try{c.onreadystatechange=d}catch(e){}}}
function lh(a){a.F&&a.ha&&(a.F.ontimeout=null);a.H&&(y.clearTimeout(a.H),a.H=null)}
k.isActive=function(){return!!this.F};
k.isComplete=function(){return 4==qh(this)};
function rh(a){var b=a.getStatus();a:switch(b){case 200:case 201:case 202:case 204:case 206:case 304:case 1223:var c=!0;break a;default:c=!1}if(!c){if(b=0===b)a=vc(1,String(a.aa)),!a&&y.self&&y.self.location&&(a=y.self.location.protocol.slice(0,-1)),b=!gh.test(a?a.toLowerCase():"");c=b}return c}
function qh(a){return a.F?a.F.readyState:0}
k.getStatus=function(){try{return 2<qh(this)?this.F.status:-1}catch(a){return-1}};
k.getLastError=function(){return"string"===typeof this.m?this.m:String(this.m)};function sh(a){I.call(this,a)}
v(sh,I);function th(a){I.call(this,a,-1,uh)}
v(th,I);var uh=[1];function ce(a){I.call(this,a)}
v(ce,I);var vh=["platform","platformVersion","architecture","model","uaFullVersion"];new th;function oe(a){I.call(this,a)}
v(oe,I);function wh(a){I.call(this,a,33,xh)}
v(wh,I);var xh=[3,20,27];function yh(a){I.call(this,a,17,zh)}
v(yh,I);var zh=[3,5];function Ah(a){I.call(this,a,6,Bh)}
v(Ah,I);var Bh=[5];function Ch(a){I.call(this,a)}
v(Ch,I);var Fh;Fh=new function(a,b,c){this.h=a;this.fieldName=b;this.j=c;this.isRepeated=0;this.i=ge;this.defaultValue=void 0}(175237375,{xg:0},Ch);function Gh(a,b,c,d,e,f,g,h,l,m,n){tf.call(this);var q=this;this.Y="";this.j=[];this.Yc="";this.Zc=this.qb=-1;this.Xb=!1;this.T=this.m=null;this.H=this.M=0;this.Zd=1;this.timeoutMillis=0;this.eb=!1;tf.call(this);this.Xc=b||function(){};
this.s=new Hh(a,f);this.Xd=d;this.Wb=n;this.ae=Xa(yf,0,1);this.ha=e||null;this.V=c||null;this.ia=g||!1;this.pageId=l||null;this.logger=null;this.withCredentials=!h;this.Vb=f||!1;!this.Vb&&(65<=jc("Chromium")||45<=jc("Firefox")||12<=jc("Safari")||(Nc()||D("iPad")||D("iPod"))&&Oc());a=ne();Ih(this.s,a);this.v=new Ug(1E4);this.i=new fg(this.v.getValue());Ke(this,this.i);m=Jh(this,m);df(this.i,"tick",m,!1,this);this.aa=new fg(6E5);Ke(this,this.aa);df(this.aa,"tick",m,!1,this);this.ia||this.aa.start();
this.Vb||(df(document,"visibilitychange",function(){"hidden"===document.visibilityState&&q.ma()}),df(document,"pagehide",this.ma,!1,this))}
v(Gh,tf);function Jh(a,b){return b?function(){b().then(function(){a.flush()})}:function(){a.flush()}}
Gh.prototype.L=function(){this.ma();tf.prototype.L.call(this)};
function Kh(a){a.ha||(a.ha=.01>a.ae()?"https://www.google.com/log?format=json&hasfast=true":"https://play.google.com/log?format=json&hasfast=true");return a.ha}
function Lh(a,b){a.v=new Ug(1>b?1:b);gg(a.i,a.v.getValue())}
Gh.prototype.log=function(a){a=a.clone();var b=this.Zd++;G(a,21,b);this.Y&&G(a,26,this.Y);if(!Zd(a,1)){b=a;var c=Date.now().toString();G(b,1,c)}null==Zd(a,15)&&G(a,15,60*(new Date).getTimezoneOffset());this.m&&(b=this.m.clone(),H(a,Sg,16,b));for(;1E3<=this.j.length;)this.j.shift(),++this.M;this.j.push(a);uf(this,new Mh(a));this.ia||this.i.enabled||this.i.start()};
Gh.prototype.flush=function(a,b){var c=this;if(0===this.j.length)a&&a();else if(this.eb)Nh(this.s,3),Oh(this);else{var d=Date.now();if(this.Zc>d&&this.qb<d)b&&b("throttled");else{Nh(this.s,1);var e=Ph(this.s,this.j,this.M,this.H);d={};var f=this.Xc();f&&(d.Authorization=f);var g=Kh(this);this.V&&(d["X-Goog-AuthUser"]=this.V,g=Ec(g,"authuser",this.V));this.pageId&&(d["X-Goog-PageId"]=this.pageId,g=Ec(g,"pageId",this.pageId));if(f&&this.Yc===f)b&&b("stale-auth-token");else{this.j=[];this.i.enabled&&
this.i.stop();this.M=0;var h=Ce(e),l;this.T&&this.T.isSupported(h.length)&&(l=this.T.compress(h));var m={url:g,body:h,ee:1,Oc:d,requestType:"POST",withCredentials:this.withCredentials,timeoutMillis:this.timeoutMillis},n=function(t){c.v.reset();gg(c.i,c.v.getValue());if(t){var A=null;try{var C=JSON.parse(t.replace(")]}'\n",""));A=new Ah(C)}catch(E){}A&&(t=Number(pe(Zd(A,1),"-1")),0<t&&(c.qb=Date.now(),c.Zc=c.qb+t),A=Fh.j?Fh.i(A,Fh.j,Fh.h,!0):Fh.isRepeated?Fh.i(A,Fh.h,!0):Fh.i(A,Fh.h,Fh.defaultValue,
!0))&&(A=pe(Zd(A,1),-1),-1!=A&&(c.Xb||Lh(c,A)))}a&&a();c.H=0},q=function(t,A){var C=F(e.P),E=!!(C&2);
C=ie(e,wh,3,E?1:2,C);var M=de(e,3,3,void 0,E);if(!(E||F(M)&8)){for(E=0;E<C.length;E++){var Q=C[E],S=he(Q);Q!==S&&(C[E]=S,M[E]=S.P)}Ed(M,8)}M=c.v;M.h=Math.min(3E5,2*M.h);M.i=Math.min(3E5,M.h+Math.round(.2*(Math.random()-.5)*M.h));gg(c.i,c.v.getValue());401===t&&f&&(c.Yc=f);void 0===A&&(A=500<=t&&600>t||401===t||0===t);A&&(c.j=C.concat(c.j),c.ia||c.i.enabled||c.i.start());b&&b("net-send-failed",t);++c.H},u=function(){c.Wb?c.Wb.send(m,n,q):c.Xd(m,n,q)};
l?l.then(function(t){m.Oc["Content-Encoding"]="gzip";m.Oc["Content-Type"]="application/binary";m.body=t;m.ee=2;u()},function(){u()}):u()}}}};
Gh.prototype.ma=function(){Qh(this.s,!0);this.flush();Qh(this.s,!1)};
function Oh(a){Rh(a,function(b,c){b=Ec(b,"format","json");var d=!1;try{d=window.navigator.sendBeacon(b,Ce(c))}catch(e){}a.eb&&!d&&(a.eb=!1);return d})}
function Rh(a,b){if(0!==a.j.length){var c=Ic(Kh(a),"format");c=Cc(c,"auth",a.Xc(),"authuser",a.V||"0");for(var d=0;10>d&&a.j.length;++d){var e=a.j.slice(0,32),f=Ph(a.s,e,a.M,a.H);if(!b(c,f)){++a.H;break}a.M=0;a.H=0;a.j=a.j.slice(e.length)}a.i.enabled&&a.i.stop()}}
function Mh(){Me.call(this,"event-logged",void 0)}
v(Mh,Me);function Hh(a,b){this.i=b=void 0===b?!1:b;this.uach=this.locale=null;this.h=new yh;G(this.h,2,a);b||(this.locale=document.documentElement.getAttribute("lang"));Ih(this,new oe)}
function Ih(a,b){H(a.h,oe,1,b);Zd(b,1)||G(b,1,1);a.i||(b=Sh(a),Zd(b,5)||G(b,5,a.locale));a.uach&&(b=Sh(a),ge(b,th,9)||H(b,th,9,a.uach))}
function Nh(a,b){ae(ge(a.h,oe,1))&&(a=Th(a),G(a,1,b))}
function Qh(a,b){ae(ge(a.h,oe,1))&&(a=Th(a),G(a,2,b))}
function Uh(a,b){var c=void 0===c?vh:c;b(window,c).then(function(d){a.uach=d;d=Sh(a);H(d,th,9,a.uach);return!0}).catch(function(){return!1})}
function Sh(a){a=ge(a.h,oe,1);var b=ge(a,ce,11);b||(b=new ce,H(a,ce,11,b));return b}
function Th(a){a=Sh(a);var b=ge(a,sh,10);b||(b=new sh,G(b,2,!1),H(a,sh,10,b));return b}
function Ph(a,b,c,d){c=void 0===c?0:c;d=void 0===d?0:d;if(ae(ge(a.h,oe,1))){var e=Th(a);me(e,3,d)}a=a.h.clone();d=Date.now().toString();a=G(a,4,d);b=ke(a,wh,3,b);c&&G(b,14,c);return b}
;function Vh(a,b,c){jh(a.url,function(d){d=d.target;if(rh(d)){try{var e=d.F?d.F.responseText:""}catch(f){e=""}b(e)}else c(d.getStatus())},a.requestType,a.body,a.Oc,a.timeoutMillis,a.withCredentials)}
;function Wh(a,b){K.call(this);this.v=a;this.T=b;this.l="https://play.google.com/log?format=json&hasfast=true";this.m=!1;this.H=Vh;this.i=""}
Ya(Wh,K);function Xh(a,b,c,d,e,f){a=void 0===a?-1:a;b=void 0===b?"":b;c=void 0===c?"":c;d=void 0===d?!1:d;e=void 0===e?"":e;K.call(this);f?a=f:(a=new Wh(a,"0"),a.i=b,Ke(this,a),""!=c&&(a.l=c),d&&(a.m=!0),e&&(a.j=e),b=new Gh(a.v,a.ha?a.ha:Rg,a.T,a.H,a.l,a.m,!1,a.Oa,void 0,void 0,a.M?a.M:void 0),Ke(a,b),a.Y&&Ih(b.s,a.Y),a.j&&(c=a.j,d=Sh(b.s),G(d,7,c)),a.s&&(b.T=a.s),a.i&&(b.Y=a.i),a.V&&((c=a.V)?(b.m||(b.m=new Sg),c=Ce(c),G(b.m,4,c)):b.m&&G(b.m,4,void 0,!1)),a.ia&&(d=a.ia,b.m||(b.m=new Sg),c=b.m,d=null==d?Rd:
Fd(d,1),G(c,2,d)),a.aa&&(c=a.aa,b.Xb=!0,Lh(b,c)),a.ma&&Uh(b.s,a.ma),a=b);this.i=a}
v(Xh,K);
Xh.prototype.flush=function(a){var b=a||[];if(b.length){a=new Dg;for(var c=[],d=0;d<b.length;d++){var e=b[d],f=e,g=new Bg;var h=G(g,1,f.i);var l=f;g=[];for(var m=0;m<l.h.length;m++)g.push(l.h[m].Kb);if(null==g)g=G(h,3,Rd);else{l=F(g);if(!(l&4)){if(l&2||Object.isFrozen(g))g=Array.prototype.slice.call(g);for(m=0;m<g.length;m++)g[m]=g[m];Gd(g,l|5)}g=G(h,3,g)}h=[];l=[];m=r(f.sb.keys());for(var n=m.next();!n.done;n=m.next())l.push(n.value.split(","));for(m=0;m<l.length;m++){n=l[m];var q=f.l;for(var u=f.yc(n)||
[],t=[],A=0;A<u.length;A++){var C=u[A];C=C&&C.gd;var E=new xg;switch(q){case 3:ee(E,1,yg,Number(C));break;case 2:ee(E,2,yg,Ud(Number(C)))}t.push(E)}q=t;for(u=0;u<q.length;u++){t=q[u];A=new zg;t=H(A,xg,2,t);A=n;C=[];E=f;for(var M=[],Q=0;Q<E.h.length;Q++)M.push(E.h[Q].Lb);E=M;for(M=0;M<E.length;M++){var S=E[M],V=A[M];Q=new vg;switch(S){case 3:ee(Q,1,wg,String(V));break;case 2:S=Q;V=Number(V);Na(V);ee(S,2,wg,V);break;case 1:ee(Q,3,wg,"true"==V)}C.push(Q)}ke(t,vg,1,C);h.push(t)}}ke(g,zg,4,h);c.push(g);
e.clear()}ke(a,Bg,1,c);b=this.i;a instanceof wh?b.log(a):(c=new wh,a=Ce(a),a=G(c,8,a),b.log(a));this.i.flush()}};function Yh(a){this.v=Zh();this.m=new Xh(1828);this.h=new ig(this.m);this.s=new og(this.h);this.j=new pg(this.h);this.l=new qg(this.h);this.i=new mg(this.h);this.Ea=sg(a)}
function Zh(){var a,b,c;return null!=(c=null==(a=globalThis.performance)?void 0:null==(b=a.now)?void 0:b.call(a))?c:Date.now()}
;function $h(){var a=this;this.promise=new Promise(function(b,c){a.resolve=b;a.reject=c})}
;function ai(a){function b(A,C){Promise.resolve().then(function(){var E;if(null!=(E=c.ja)){var M=Zh()-E.v;E.s.h.vc("/client_streamz/bg/fil",M,E.Ea)}g.resolve({ce:A,ef:C})})}
var c=this;this.i=!1;var d=a.program;var e=a.ue;if(a.Oe){var f;this.ja=null!=(f=a.ja)?f:new Yh(e)}var g=new $h;this.j=g.promise;if(y[e])if(y[e].a){var h;null!=(h=this.ja)&&ng(h.i,h.Ea,3)}else{var l;null!=(l=this.ja)&&ng(l.i,l.Ea,2);var m;null!=(m=this.ja)&&m.h.pb()}else{var n;null!=(n=this.ja)&&ng(n.i,n.Ea,1);var q;null!=(q=this.ja)&&q.h.pb()}try{this.l=r((0,y[e].a)(d,b,!0)).next().value,this.df=g.promise.then(function(){})}catch(A){var u;
null!=(u=this.ja)&&ng(u.i,u.Ea,4);var t;null!=(t=this.ja)&&t.h.pb();throw A;}}
ai.prototype.snapshot=function(a){var b=this;if(this.i)throw Error("Already disposed");var c=Zh(),d;null!=(d=this.ja)&&d.j.h.uc("/client_streamz/bg/fsc",d.Ea);return this.j.then(function(e){var f=e.ce;return new Promise(function(g){f(function(h){var l;if(null!=(l=b.ja)){var m=Zh()-c;l.l.h.vc("/client_streamz/bg/fsl",m,l.Ea)}g(h)},[a.jd,
a.ff])})})};
ai.prototype.Md=function(a){if(this.i)throw Error("Already disposed");var b=Zh(),c;null!=(c=this.ja)&&c.j.h.uc("/client_streamz/bg/fsc",c.Ea);a=this.l([a.jd,a.ff]);null!=(c=this.ja)&&(b=Zh()-b,c.l.h.vc("/client_streamz/bg/fsl",b,c.Ea));return a};
ai.prototype.dispose=function(){var a;null!=(a=this.ja)&&a.h.pb();this.i=!0;this.j.then(function(b){(b=b.ef)&&b()})};
ai.prototype.h=function(){return this.i};var bi=window;Bb("csi.gstatic.com");Bb("googleads.g.doubleclick.net");Bb("partner.googleadservices.com");Bb("pubads.g.doubleclick.net");Bb("securepubads.g.doubleclick.net");Bb("tpc.googlesyndication.com");/*

 SPDX-License-Identifier: Apache-2.0
*/
var ci;try{new URL("s://g"),ci=!0}catch(a){ci=!1}var di=ci;var ei={};function fi(){}
function gi(a){this.h=a}
v(gi,fi);gi.prototype.toString=function(){return this.h};function hi(a){var b="true".toString(),c=[new gi(ii[0].toLowerCase(),ei)];if(0===c.length)throw Error("");if(c.map(function(d){if(d instanceof gi)d=d.h;else throw Error("");return d}).every(function(d){return 0!=="data-loaded".indexOf(d)}))throw Error('Attribute "data-loaded" does not match any of the allowed prefixes.');
a.setAttribute("data-loaded",b)}
;function ji(a){var b,c,d=null==(c=(b=(a.ownerDocument&&a.ownerDocument.defaultView||window).document).querySelector)?void 0:c.call(b,"script[nonce]");(b=d?d.nonce||d.getAttribute("nonce")||"":"")&&a.setAttribute("nonce",b)}
function ki(a,b){a.src=Gb(b);ji(a)}
;function li(a){this.Ce=a}
function mi(a){return new li(function(b){return b.substr(0,a.length+1).toLowerCase()===a+":"})}
var ni=[mi("data"),mi("http"),mi("https"),mi("mailto"),mi("ftp"),new li(function(a){return/^[^:]*([/?#]|$)/.test(a)})];function oi(a){var b=pi;if(b)for(var c in b)Object.prototype.hasOwnProperty.call(b,c)&&a(b[c],c,b)}
function qi(){var a=[];oi(function(b){a.push(b)});
return a}
var pi={pf:"allow-forms",qf:"allow-modals",rf:"allow-orientation-lock",sf:"allow-pointer-lock",tf:"allow-popups",uf:"allow-popups-to-escape-sandbox",vf:"allow-presentation",wf:"allow-same-origin",xf:"allow-scripts",yf:"allow-top-navigation",zf:"allow-top-navigation-by-user-activation"},ri=db(function(){return qi()});
function si(){var a=ti(),b={};fb(ri(),function(c){a.sandbox&&a.sandbox.supports&&a.sandbox.supports(c)&&(b[c]=!0)});
return b}
function ti(){var a=void 0===a?document:a;return a.createElement("iframe")}
;function ui(a){"number"==typeof a&&(a=Math.round(a)+"px");return a}
;var vi=(new Date).getTime();var wi="client_dev_domain client_dev_regex_map client_dev_root_url client_rollout_override expflag forcedCapability jsfeat jsmode mods".split(" ");ha(wi);"undefined"!==typeof TextDecoder&&new TextDecoder;var xi="undefined"!==typeof TextEncoder?new TextEncoder:null,yi=xi?function(a){return xi.encode(a)}:function(a){a=Kc(a);
for(var b=new Uint8Array(a.length),c=0;c<b.length;c++)b[c]=a[c];return b};function zi(a){tf.call(this);var b=this;this.v=this.j=0;this.ya=null!=a?a:{ea:function(e,f){return setTimeout(e,f)},
Ba:function(e){clearTimeout(e)}};
var c,d;this.i=null!=(d=null==(c=window.navigator)?void 0:c.onLine)?d:!0;this.m=function(){return x(function(e){return w(e,Ai(b),0)})};
window.addEventListener("offline",this.m);window.addEventListener("online",this.m);this.v||Bi(this)}
v(zi,tf);function Ci(){var a=Di;zi.h||(zi.h=new zi(a));return zi.h}
zi.prototype.dispose=function(){window.removeEventListener("offline",this.m);window.removeEventListener("online",this.m);this.ya.Ba(this.v);delete zi.h};
zi.prototype.oa=function(){return this.i};
function Bi(a){a.v=a.ya.ea(function(){var b;return x(function(c){if(1==c.h)return a.i?(null==(b=window.navigator)?0:b.onLine)?c.A(3):w(c,Ai(a),3):w(c,Ai(a),3);Bi(a);c.h=0})},3E4)}
function Ai(a,b){return a.s?a.s:a.s=new Promise(function(c){var d,e,f,g;return x(function(h){switch(h.h){case 1:return d=window.AbortController?new window.AbortController:void 0,f=null==(e=d)?void 0:e.signal,g=!1,xa(h,2,3),d&&(a.j=a.ya.ea(function(){d.abort()},b||2E4)),w(h,fetch("/generate_204",{method:"HEAD",
signal:f}),5);case 5:g=!0;case 3:Aa(h);a.s=void 0;a.j&&(a.ya.Ba(a.j),a.j=0);g!==a.i&&(a.i=g,a.i?uf(a,"networkstatus-online"):uf(a,"networkstatus-offline"));c(g);Ba(h);break;case 2:za(h),g=!1,h.A(3)}})})}
;function Ei(){this.data_=[];this.h=-1}
Ei.prototype.set=function(a,b){b=void 0===b?!0:b;0<=a&&52>a&&Number.isInteger(a)&&this.data_[a]!==b&&(this.data_[a]=b,this.h=-1)};
Ei.prototype.get=function(a){return!!this.data_[a]};
function Fi(a){-1===a.h&&(a.h=ib(a.data_,function(b,c,d){return c?b+Math.pow(2,d):b},0));
return a.h}
;function Gi(a,b){this.h=a[y.Symbol.iterator]();this.i=b}
Gi.prototype[Symbol.iterator]=function(){return this};
Gi.prototype.next=function(){var a=this.h.next();return{value:a.done?void 0:this.i.call(void 0,a.value),done:a.done}};
function Hi(a,b){return new Gi(a,b)}
;function Ii(){this.blockSize=-1}
;function Ji(){this.blockSize=-1;this.blockSize=64;this.h=[];this.m=[];this.s=[];this.j=[];this.j[0]=128;for(var a=1;a<this.blockSize;++a)this.j[a]=0;this.l=this.i=0;this.reset()}
Ya(Ji,Ii);Ji.prototype.reset=function(){this.h[0]=1732584193;this.h[1]=4023233417;this.h[2]=2562383102;this.h[3]=271733878;this.h[4]=3285377520;this.l=this.i=0};
function Ki(a,b,c){c||(c=0);var d=a.s;if("string"===typeof b)for(var e=0;16>e;e++)d[e]=b.charCodeAt(c)<<24|b.charCodeAt(c+1)<<16|b.charCodeAt(c+2)<<8|b.charCodeAt(c+3),c+=4;else for(e=0;16>e;e++)d[e]=b[c]<<24|b[c+1]<<16|b[c+2]<<8|b[c+3],c+=4;for(e=16;80>e;e++){var f=d[e-3]^d[e-8]^d[e-14]^d[e-16];d[e]=(f<<1|f>>>31)&4294967295}b=a.h[0];c=a.h[1];var g=a.h[2],h=a.h[3],l=a.h[4];for(e=0;80>e;e++){if(40>e)if(20>e){f=h^c&(g^h);var m=1518500249}else f=c^g^h,m=1859775393;else 60>e?(f=c&g|h&(c|g),m=2400959708):
(f=c^g^h,m=3395469782);f=(b<<5|b>>>27)+f+l+m+d[e]&4294967295;l=h;h=g;g=(c<<30|c>>>2)&4294967295;c=b;b=f}a.h[0]=a.h[0]+b&4294967295;a.h[1]=a.h[1]+c&4294967295;a.h[2]=a.h[2]+g&4294967295;a.h[3]=a.h[3]+h&4294967295;a.h[4]=a.h[4]+l&4294967295}
Ji.prototype.update=function(a,b){if(null!=a){void 0===b&&(b=a.length);for(var c=b-this.blockSize,d=0,e=this.m,f=this.i;d<b;){if(0==f)for(;d<=c;)Ki(this,a,d),d+=this.blockSize;if("string"===typeof a)for(;d<b;){if(e[f]=a.charCodeAt(d),++f,++d,f==this.blockSize){Ki(this,e);f=0;break}}else for(;d<b;)if(e[f]=a[d],++f,++d,f==this.blockSize){Ki(this,e);f=0;break}}this.i=f;this.l+=b}};
Ji.prototype.digest=function(){var a=[],b=8*this.l;56>this.i?this.update(this.j,56-this.i):this.update(this.j,this.blockSize-(this.i-56));for(var c=this.blockSize-1;56<=c;c--)this.m[c]=b&255,b/=256;Ki(this,this.m);for(c=b=0;5>c;c++)for(var d=24;0<=d;d-=8)a[b]=this.h[c]>>d&255,++b;return a};function Li(a){return"string"==typeof a.className?a.className:a.getAttribute&&a.getAttribute("class")||""}
function Mi(a,b){"string"==typeof a.className?a.className=b:a.setAttribute&&a.setAttribute("class",b)}
function Ni(a,b){a.classList?b=a.classList.contains(b):(a=a.classList?a.classList:Li(a).match(/\S+/g)||[],b=0<=eb(a,b));return b}
function Oi(){var a=document.body;a.classList?a.classList.remove("inverted-hdpi"):Ni(a,"inverted-hdpi")&&Mi(a,Array.prototype.filter.call(a.classList?a.classList:Li(a).match(/\S+/g)||[],function(b){return"inverted-hdpi"!=b}).join(" "))}
;function Pi(){}
Pi.prototype.next=function(){return Qi};
var Qi={done:!0,value:void 0};function Ri(a){return{value:a,done:!1}}
Pi.prototype.Aa=function(){return this};function Si(a){if(a instanceof Ti||a instanceof Ui||a instanceof Vi)return a;if("function"==typeof a.next)return new Ti(function(){return a});
if("function"==typeof a[Symbol.iterator])return new Ti(function(){return a[Symbol.iterator]()});
if("function"==typeof a.Aa)return new Ti(function(){return a.Aa()});
throw Error("Not an iterator or iterable.");}
function Ti(a){this.i=a}
Ti.prototype.Aa=function(){return new Ui(this.i())};
Ti.prototype[Symbol.iterator]=function(){return new Vi(this.i())};
Ti.prototype.h=function(){return new Vi(this.i())};
function Ui(a){this.i=a}
v(Ui,Pi);Ui.prototype.next=function(){return this.i.next()};
Ui.prototype[Symbol.iterator]=function(){return new Vi(this.i)};
Ui.prototype.h=function(){return new Vi(this.i)};
function Vi(a){Ti.call(this,function(){return a});
this.j=a}
v(Vi,Ti);Vi.prototype.next=function(){return this.j.next()};function Wi(a,b){this.i={};this.h=[];this.Va=this.size=0;var c=arguments.length;if(1<c){if(c%2)throw Error("Uneven number of arguments");for(var d=0;d<c;d+=2)this.set(arguments[d],arguments[d+1])}else if(a)if(a instanceof Wi)for(c=a.Bc(),d=0;d<c.length;d++)this.set(c[d],a.get(c[d]));else for(d in a)this.set(d,a[d])}
k=Wi.prototype;k.Bc=function(){Xi(this);return this.h.concat()};
k.has=function(a){return Yi(this.i,a)};
k.equals=function(a,b){if(this===a)return!0;if(this.size!=a.size)return!1;b=b||Zi;Xi(this);for(var c,d=0;c=this.h[d];d++)if(!b(this.get(c),a.get(c)))return!1;return!0};
function Zi(a,b){return a===b}
k.clear=function(){this.i={};this.Va=this.size=this.h.length=0};
k.remove=function(a){return this.delete(a)};
k.delete=function(a){return Yi(this.i,a)?(delete this.i[a],--this.size,this.Va++,this.h.length>2*this.size&&Xi(this),!0):!1};
function Xi(a){if(a.size!=a.h.length){for(var b=0,c=0;b<a.h.length;){var d=a.h[b];Yi(a.i,d)&&(a.h[c++]=d);b++}a.h.length=c}if(a.size!=a.h.length){var e={};for(c=b=0;b<a.h.length;)d=a.h[b],Yi(e,d)||(a.h[c++]=d,e[d]=1),b++;a.h.length=c}}
k.get=function(a,b){return Yi(this.i,a)?this.i[a]:b};
k.set=function(a,b){Yi(this.i,a)||(this.size+=1,this.h.push(a),this.Va++);this.i[a]=b};
k.forEach=function(a,b){for(var c=this.Bc(),d=0;d<c.length;d++){var e=c[d],f=this.get(e);a.call(b,f,e,this)}};
k.clone=function(){return new Wi(this)};
k.keys=function(){return Si(this.Aa(!0)).h()};
k.values=function(){return Si(this.Aa(!1)).h()};
k.entries=function(){var a=this;return Hi(this.keys(),function(b){return[b,a.get(b)]})};
k.Aa=function(a){Xi(this);var b=0,c=this.Va,d=this,e=new Pi;e.next=function(){if(c!=d.Va)throw Error("The map has changed since the iterator was created");if(b>=d.h.length)return Qi;var f=d.h[b++];return Ri(a?f:d.i[f])};
return e};
function Yi(a,b){return Object.prototype.hasOwnProperty.call(a,b)}
;function $i(a){K.call(this);this.s=1;this.l=[];this.m=0;this.i=[];this.j={};this.v=!!a}
Ya($i,K);k=$i.prototype;k.subscribe=function(a,b,c){var d=this.j[a];d||(d=this.j[a]=[]);var e=this.s;this.i[e]=a;this.i[e+1]=b;this.i[e+2]=c;this.s=e+3;d.push(e);return e};
function aj(a,b,c,d){if(b=a.j[b]){var e=a.i;(b=b.find(function(f){return e[f+1]==c&&e[f+2]==d}))&&a.Fb(b)}}
k.Fb=function(a){var b=this.i[a];if(b){var c=this.j[b];0!=this.m?(this.l.push(a),this.i[a+1]=function(){}):(c&&kb(c,a),delete this.i[a],delete this.i[a+1],delete this.i[a+2])}return!!b};
k.cb=function(a,b){var c=this.j[a];if(c){for(var d=Array(arguments.length-1),e=1,f=arguments.length;e<f;e++)d[e-1]=arguments[e];if(this.v)for(e=0;e<c.length;e++){var g=c[e];bj(this.i[g+1],this.i[g+2],d)}else{this.m++;try{for(e=0,f=c.length;e<f&&!this.h();e++)g=c[e],this.i[g+1].apply(this.i[g+2],d)}finally{if(this.m--,0<this.l.length&&0==this.m)for(;c=this.l.pop();)this.Fb(c)}}return 0!=e}return!1};
function bj(a,b,c){Mf(function(){a.apply(b,c)})}
k.clear=function(a){if(a){var b=this.j[a];b&&(b.forEach(this.Fb,this),delete this.j[a])}else this.i.length=0,this.j={}};
k.L=function(){$i.va.L.call(this);this.clear();this.l.length=0};function cj(a){this.h=a}
cj.prototype.set=function(a,b){void 0===b?this.h.remove(a):this.h.set(a,Vg(b))};
cj.prototype.get=function(a){try{var b=this.h.get(a)}catch(c){return}if(null!==b)try{return JSON.parse(b)}catch(c){throw"Storage: Invalid value was encountered";}};
cj.prototype.remove=function(a){this.h.remove(a)};function dj(a){this.h=a}
Ya(dj,cj);function ej(a){this.data=a}
function fj(a){return void 0===a||a instanceof ej?a:new ej(a)}
dj.prototype.set=function(a,b){dj.va.set.call(this,a,fj(b))};
dj.prototype.i=function(a){a=dj.va.get.call(this,a);if(void 0===a||a instanceof Object)return a;throw"Storage: Invalid value was encountered";};
dj.prototype.get=function(a){if(a=this.i(a)){if(a=a.data,void 0===a)throw"Storage: Invalid value was encountered";}else a=void 0;return a};function gj(a){this.h=a}
Ya(gj,dj);gj.prototype.set=function(a,b,c){if(b=fj(b)){if(c){if(c<Date.now()){gj.prototype.remove.call(this,a);return}b.expiration=c}b.creation=Date.now()}gj.va.set.call(this,a,b)};
gj.prototype.i=function(a){var b=gj.va.i.call(this,a);if(b){var c=b.creation,d=b.expiration;if(d&&d<Date.now()||c&&c>Date.now())gj.prototype.remove.call(this,a);else return b}};function hj(){}
;function ij(){}
Ya(ij,hj);ij.prototype[Symbol.iterator]=function(){return Si(this.Aa(!0)).h()};
ij.prototype.clear=function(){var a=Array.from(this);a=r(a);for(var b=a.next();!b.done;b=a.next())this.remove(b.value)};function jj(a){this.h=a}
Ya(jj,ij);k=jj.prototype;k.isAvailable=function(){if(!this.h)return!1;try{return this.h.setItem("__sak","1"),this.h.removeItem("__sak"),!0}catch(a){return!1}};
k.set=function(a,b){try{this.h.setItem(a,b)}catch(c){if(0==this.h.length)throw"Storage mechanism: Storage disabled";throw"Storage mechanism: Quota exceeded";}};
k.get=function(a){a=this.h.getItem(a);if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
k.remove=function(a){this.h.removeItem(a)};
k.Aa=function(a){var b=0,c=this.h,d=new Pi;d.next=function(){if(b>=c.length)return Qi;var e=c.key(b++);if(a)return Ri(e);e=c.getItem(e);if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return Ri(e)};
return d};
k.clear=function(){this.h.clear()};
k.key=function(a){return this.h.key(a)};function kj(){var a=null;try{a=window.localStorage||null}catch(b){}this.h=a}
Ya(kj,jj);function lj(a,b){this.i=a;this.h=null;var c;if(c=Wc)c=!(9<=Number(id));if(c){mj||(mj=new Wi);this.h=mj.get(a);this.h||(b?this.h=document.getElementById(b):(this.h=document.createElement("userdata"),this.h.addBehavior("#default#userData"),document.body.appendChild(this.h)),mj.set(a,this.h));try{this.h.load(this.i)}catch(d){this.h=null}}}
Ya(lj,ij);var nj={".":".2E","!":".21","~":".7E","*":".2A","'":".27","(":".28",")":".29","%":"."},mj=null;function oj(a){return"_"+encodeURIComponent(a).replace(/[.!~*'()%]/g,function(b){return nj[b]})}
k=lj.prototype;k.isAvailable=function(){return!!this.h};
k.set=function(a,b){this.h.setAttribute(oj(a),b);pj(this)};
k.get=function(a){a=this.h.getAttribute(oj(a));if("string"!==typeof a&&null!==a)throw"Storage mechanism: Invalid value was encountered";return a};
k.remove=function(a){this.h.removeAttribute(oj(a));pj(this)};
k.Aa=function(a){var b=0,c=this.h.XMLDocument.documentElement.attributes,d=new Pi;d.next=function(){if(b>=c.length)return Qi;var e=c[b++];if(a)return Ri(decodeURIComponent(e.nodeName.replace(/\./g,"%")).slice(1));e=e.nodeValue;if("string"!==typeof e)throw"Storage mechanism: Invalid value was encountered";return Ri(e)};
return d};
k.clear=function(){for(var a=this.h.XMLDocument.documentElement,b=a.attributes.length;0<b;b--)a.removeAttribute(a.attributes[b-1].nodeName);pj(this)};
function pj(a){try{a.h.save(a.i)}catch(b){throw"Storage mechanism: Quota exceeded";}}
;function qj(a,b){this.i=a;this.h=b+"::"}
Ya(qj,ij);qj.prototype.set=function(a,b){this.i.set(this.h+a,b)};
qj.prototype.get=function(a){return this.i.get(this.h+a)};
qj.prototype.remove=function(a){this.i.remove(this.h+a)};
qj.prototype.Aa=function(a){var b=this.i[Symbol.iterator](),c=this,d=new Pi;d.next=function(){var e=b.next();if(e.done)return e;for(e=e.value;e.slice(0,c.h.length)!=c.h;){e=b.next();if(e.done)return e;e=e.value}return Ri(a?e.slice(c.h.length):c.i.get(e))};
return d};/*

 (The MIT License)

 Copyright (C) 2014 by Vitaly Puzrin

 Permission is hereby granted, free of charge, to any person obtaining a copy
 of this software and associated documentation files (the "Software"), to deal
 in the Software without restriction, including without limitation the rights
 to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 copies of the Software, and to permit persons to whom the Software is
 furnished to do so, subject to the following conditions:

 The above copyright notice and this permission notice shall be included in
 all copies or substantial portions of the Software.

 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE.

 -----------------------------------------------------------------------------
 Ported from zlib, which is under the following license
 https://github.com/madler/zlib/blob/master/zlib.h

 zlib.h -- interface of the 'zlib' general purpose compression library
   version 1.2.8, April 28th, 2013
   Copyright (C) 1995-2013 Jean-loup Gailly and Mark Adler
   This software is provided 'as-is', without any express or implied
   warranty.  In no event will the authors be held liable for any damages
   arising from the use of this software.
   Permission is granted to anyone to use this software for any purpose,
   including commercial applications, and to alter it and redistribute it
   freely, subject to the following restrictions:
   1. The origin of this software must not be misrepresented; you must not
      claim that you wrote the original software. If you use this software
      in a product, an acknowledgment in the product documentation would be
      appreciated but is not required.
   2. Altered source versions must be plainly marked as such, and must not be
      misrepresented as being the original software.
   3. This notice may not be removed or altered from any source distribution.
   Jean-loup Gailly        Mark Adler
   jloup@gzip.org          madler@alumni.caltech.edu
   The data format used by the zlib library is described by RFCs (Request for
   Comments) 1950 to 1952 in the files http://tools.ietf.org/html/rfc1950
   (zlib format), rfc1951 (deflate format) and rfc1952 (gzip format).
*/
var L={},rj="undefined"!==typeof Uint8Array&&"undefined"!==typeof Uint16Array&&"undefined"!==typeof Int32Array;L.assign=function(a){for(var b=Array.prototype.slice.call(arguments,1);b.length;){var c=b.shift();if(c){if("object"!==typeof c)throw new TypeError(c+"must be non-object");for(var d in c)Object.prototype.hasOwnProperty.call(c,d)&&(a[d]=c[d])}}return a};
L.Rc=function(a,b){if(a.length===b)return a;if(a.subarray)return a.subarray(0,b);a.length=b;return a};
var sj={rb:function(a,b,c,d,e){if(b.subarray&&a.subarray)a.set(b.subarray(c,c+d),e);else for(var f=0;f<d;f++)a[e+f]=b[c+f]},
od:function(a){var b,c;var d=c=0;for(b=a.length;d<b;d++)c+=a[d].length;var e=new Uint8Array(c);d=c=0;for(b=a.length;d<b;d++){var f=a[d];e.set(f,c);c+=f.length}return e}},tj={rb:function(a,b,c,d,e){for(var f=0;f<d;f++)a[e+f]=b[c+f]},
od:function(a){return[].concat.apply([],a)}};
L.cf=function(){rj?(L.bb=Uint8Array,L.Fa=Uint16Array,L.Vd=Int32Array,L.assign(L,sj)):(L.bb=Array,L.Fa=Array,L.Vd=Array,L.assign(L,tj))};
L.cf();var uj=!0;try{new Uint8Array(1)}catch(a){uj=!1}for(var vj=new L.bb(256),wj=0;256>wj;wj++)vj[wj]=252<=wj?6:248<=wj?5:240<=wj?4:224<=wj?3:192<=wj?2:1;vj[254]=vj[254]=1;
function xj(a){var b,c,d=a.length,e=0;for(b=0;b<d;b++){var f=a.charCodeAt(b);if(55296===(f&64512)&&b+1<d){var g=a.charCodeAt(b+1);56320===(g&64512)&&(f=65536+(f-55296<<10)+(g-56320),b++)}e+=128>f?1:2048>f?2:65536>f?3:4}var h=new L.bb(e);for(b=c=0;c<e;b++)f=a.charCodeAt(b),55296===(f&64512)&&b+1<d&&(g=a.charCodeAt(b+1),56320===(g&64512)&&(f=65536+(f-55296<<10)+(g-56320),b++)),128>f?h[c++]=f:(2048>f?h[c++]=192|f>>>6:(65536>f?h[c++]=224|f>>>12:(h[c++]=240|f>>>18,h[c++]=128|f>>>12&63),h[c++]=128|f>>>
6&63),h[c++]=128|f&63);return h}
;var yj={};yj=function(a,b,c,d){var e=a&65535|0;a=a>>>16&65535|0;for(var f;0!==c;){f=2E3<c?2E3:c;c-=f;do e=e+b[d++]|0,a=a+e|0;while(--f);e%=65521;a%=65521}return e|a<<16|0};for(var zj={},Aj,Bj=[],Cj=0;256>Cj;Cj++){Aj=Cj;for(var Dj=0;8>Dj;Dj++)Aj=Aj&1?3988292384^Aj>>>1:Aj>>>1;Bj[Cj]=Aj}zj=function(a,b,c,d){c=d+c;for(a^=-1;d<c;d++)a=a>>>8^Bj[(a^b[d])&255];return a^-1};var Ej={};Ej={2:"need dictionary",1:"stream end",0:"","-1":"file error","-2":"stream error","-3":"data error","-4":"insufficient memory","-5":"buffer error","-6":"incompatible version"};function Fj(a){for(var b=a.length;0<=--b;)a[b]=0}
var Gj=[0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0],Hj=[0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13],Ij=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,3,7],Jj=[16,17,18,0,8,7,9,6,10,5,11,4,12,3,13,2,14,1,15],Kj=Array(576);Fj(Kj);var Lj=Array(60);Fj(Lj);var Mj=Array(512);Fj(Mj);var Nj=Array(256);Fj(Nj);var Oj=Array(29);Fj(Oj);var Pj=Array(30);Fj(Pj);function Qj(a,b,c,d,e){this.Nd=a;this.pe=b;this.oe=c;this.ke=d;this.Ge=e;this.sd=a&&a.length}
var Rj,Sj,Tj;function Uj(a,b){this.md=a;this.yb=0;this.Ua=b}
function Vj(a,b){a.R[a.pending++]=b&255;a.R[a.pending++]=b>>>8&255}
function Wj(a,b,c){a.Z>16-c?(a.ga|=b<<a.Z&65535,Vj(a,a.ga),a.ga=b>>16-a.Z,a.Z+=c-16):(a.ga|=b<<a.Z&65535,a.Z+=c)}
function Xj(a,b,c){Wj(a,c[2*b],c[2*b+1])}
function Yj(a,b){var c=0;do c|=a&1,a>>>=1,c<<=1;while(0<--b);return c>>>1}
function Zj(a,b,c){var d=Array(16),e=0,f;for(f=1;15>=f;f++)d[f]=e=e+c[f-1]<<1;for(c=0;c<=b;c++)e=a[2*c+1],0!==e&&(a[2*c]=Yj(d[e]++,e))}
function ak(a){var b;for(b=0;286>b;b++)a.ka[2*b]=0;for(b=0;30>b;b++)a.gb[2*b]=0;for(b=0;19>b;b++)a.ba[2*b]=0;a.ka[512]=1;a.Ma=a.Bb=0;a.sa=a.matches=0}
function bk(a){8<a.Z?Vj(a,a.ga):0<a.Z&&(a.R[a.pending++]=a.ga);a.ga=0;a.Z=0}
function ck(a,b,c){bk(a);Vj(a,c);Vj(a,~c);L.rb(a.R,a.window,b,c,a.pending);a.pending+=c}
function dk(a,b,c,d){var e=2*b,f=2*c;return a[e]<a[f]||a[e]===a[f]&&d[b]<=d[c]}
function ek(a,b,c){for(var d=a.S[c],e=c<<1;e<=a.Ja;){e<a.Ja&&dk(b,a.S[e+1],a.S[e],a.depth)&&e++;if(dk(b,d,a.S[e],a.depth))break;a.S[c]=a.S[e];c=e;e<<=1}a.S[c]=d}
function fk(a,b,c){var d=0;if(0!==a.sa){do{var e=a.R[a.Hb+2*d]<<8|a.R[a.Hb+2*d+1];var f=a.R[a.Gc+d];d++;if(0===e)Xj(a,f,b);else{var g=Nj[f];Xj(a,g+256+1,b);var h=Gj[g];0!==h&&(f-=Oj[g],Wj(a,f,h));e--;g=256>e?Mj[e]:Mj[256+(e>>>7)];Xj(a,g,c);h=Hj[g];0!==h&&(e-=Pj[g],Wj(a,e,h))}}while(d<a.sa)}Xj(a,256,b)}
function gk(a,b){var c=b.md,d=b.Ua.Nd,e=b.Ua.sd,f=b.Ua.ke,g,h=-1;a.Ja=0;a.vb=573;for(g=0;g<f;g++)0!==c[2*g]?(a.S[++a.Ja]=h=g,a.depth[g]=0):c[2*g+1]=0;for(;2>a.Ja;){var l=a.S[++a.Ja]=2>h?++h:0;c[2*l]=1;a.depth[l]=0;a.Ma--;e&&(a.Bb-=d[2*l+1])}b.yb=h;for(g=a.Ja>>1;1<=g;g--)ek(a,c,g);l=f;do g=a.S[1],a.S[1]=a.S[a.Ja--],ek(a,c,1),d=a.S[1],a.S[--a.vb]=g,a.S[--a.vb]=d,c[2*l]=c[2*g]+c[2*d],a.depth[l]=(a.depth[g]>=a.depth[d]?a.depth[g]:a.depth[d])+1,c[2*g+1]=c[2*d+1]=l,a.S[1]=l++,ek(a,c,1);while(2<=a.Ja);a.S[--a.vb]=
a.S[1];g=b.md;l=b.yb;d=b.Ua.Nd;e=b.Ua.sd;f=b.Ua.pe;var m=b.Ua.oe,n=b.Ua.Ge,q,u=0;for(q=0;15>=q;q++)a.Ga[q]=0;g[2*a.S[a.vb]+1]=0;for(b=a.vb+1;573>b;b++){var t=a.S[b];q=g[2*g[2*t+1]+1]+1;q>n&&(q=n,u++);g[2*t+1]=q;if(!(t>l)){a.Ga[q]++;var A=0;t>=m&&(A=f[t-m]);var C=g[2*t];a.Ma+=C*(q+A);e&&(a.Bb+=C*(d[2*t+1]+A))}}if(0!==u){do{for(q=n-1;0===a.Ga[q];)q--;a.Ga[q]--;a.Ga[q+1]+=2;a.Ga[n]--;u-=2}while(0<u);for(q=n;0!==q;q--)for(t=a.Ga[q];0!==t;)d=a.S[--b],d>l||(g[2*d+1]!==q&&(a.Ma+=(q-g[2*d+1])*g[2*d],g[2*
d+1]=q),t--)}Zj(c,h,a.Ga)}
function hk(a,b,c){var d,e=-1,f=b[1],g=0,h=7,l=4;0===f&&(h=138,l=3);b[2*(c+1)+1]=65535;for(d=0;d<=c;d++){var m=f;f=b[2*(d+1)+1];++g<h&&m===f||(g<l?a.ba[2*m]+=g:0!==m?(m!==e&&a.ba[2*m]++,a.ba[32]++):10>=g?a.ba[34]++:a.ba[36]++,g=0,e=m,0===f?(h=138,l=3):m===f?(h=6,l=3):(h=7,l=4))}}
function ik(a,b,c){var d,e=-1,f=b[1],g=0,h=7,l=4;0===f&&(h=138,l=3);for(d=0;d<=c;d++){var m=f;f=b[2*(d+1)+1];if(!(++g<h&&m===f)){if(g<l){do Xj(a,m,a.ba);while(0!==--g)}else 0!==m?(m!==e&&(Xj(a,m,a.ba),g--),Xj(a,16,a.ba),Wj(a,g-3,2)):10>=g?(Xj(a,17,a.ba),Wj(a,g-3,3)):(Xj(a,18,a.ba),Wj(a,g-11,7));g=0;e=m;0===f?(h=138,l=3):m===f?(h=6,l=3):(h=7,l=4)}}}
function jk(a){var b=4093624447,c;for(c=0;31>=c;c++,b>>>=1)if(b&1&&0!==a.ka[2*c])return 0;if(0!==a.ka[18]||0!==a.ka[20]||0!==a.ka[26])return 1;for(c=32;256>c;c++)if(0!==a.ka[2*c])return 1;return 0}
var kk=!1;function lk(a,b,c){a.R[a.Hb+2*a.sa]=b>>>8&255;a.R[a.Hb+2*a.sa+1]=b&255;a.R[a.Gc+a.sa]=c&255;a.sa++;0===b?a.ka[2*c]++:(a.matches++,b--,a.ka[2*(Nj[c]+256+1)]++,a.gb[2*(256>b?Mj[b]:Mj[256+(b>>>7)])]++);return a.sa===a.Mb-1}
;function mk(a,b){a.msg=Ej[b];return b}
function nk(a){for(var b=a.length;0<=--b;)a[b]=0}
function ok(a){var b=a.state,c=b.pending;c>a.I&&(c=a.I);0!==c&&(L.rb(a.Nb,b.R,b.Ob,c,a.zb),a.zb+=c,b.Ob+=c,a.Sc+=c,a.I-=c,b.pending-=c,0===b.pending&&(b.Ob=0))}
function pk(a,b){var c=0<=a.na?a.na:-1,d=a.o-a.na,e=0;if(0<a.level){2===a.D.wc&&(a.D.wc=jk(a));gk(a,a.jc);gk(a,a.ec);hk(a,a.ka,a.jc.yb);hk(a,a.gb,a.ec.yb);gk(a,a.dd);for(e=18;3<=e&&0===a.ba[2*Jj[e]+1];e--);a.Ma+=3*(e+1)+14;var f=a.Ma+3+7>>>3;var g=a.Bb+3+7>>>3;g<=f&&(f=g)}else f=g=d+5;if(d+4<=f&&-1!==c)Wj(a,b?1:0,3),ck(a,c,d);else if(4===a.strategy||g===f)Wj(a,2+(b?1:0),3),fk(a,Kj,Lj);else{Wj(a,4+(b?1:0),3);c=a.jc.yb+1;d=a.ec.yb+1;e+=1;Wj(a,c-257,5);Wj(a,d-1,5);Wj(a,e-4,4);for(f=0;f<e;f++)Wj(a,a.ba[2*
Jj[f]+1],3);ik(a,a.ka,c-1);ik(a,a.gb,d-1);fk(a,a.ka,a.gb)}ak(a);b&&bk(a);a.na=a.o;ok(a.D)}
function N(a,b){a.R[a.pending++]=b}
function qk(a,b){a.R[a.pending++]=b>>>8&255;a.R[a.pending++]=b&255}
function rk(a,b){var c=a.yd,d=a.o,e=a.qa,f=a.Ad,g=a.o>a.da-262?a.o-(a.da-262):0,h=a.window,l=a.Wa,m=a.Da,n=a.o+258,q=h[d+e-1],u=h[d+e];a.qa>=a.rd&&(c>>=2);f>a.u&&(f=a.u);do{var t=b;if(h[t+e]===u&&h[t+e-1]===q&&h[t]===h[d]&&h[++t]===h[d+1]){d+=2;for(t++;h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&h[++d]===h[++t]&&d<n;);t=258-(n-d);d=n-258;if(t>e){a.xb=b;e=t;if(t>=f)break;q=h[d+e-1];u=h[d+e]}}}while((b=m[b&l])>g&&0!==--c);return e<=
a.u?e:a.u}
function sk(a){var b=a.da,c;do{var d=a.Td-a.u-a.o;if(a.o>=b+(b-262)){L.rb(a.window,a.window,b,b,0);a.xb-=b;a.o-=b;a.na-=b;var e=c=a.ic;do{var f=a.head[--e];a.head[e]=f>=b?f-b:0}while(--c);e=c=b;do f=a.Da[--e],a.Da[e]=f>=b?f-b:0;while(--c);d+=b}if(0===a.D.fa)break;e=a.D;c=a.window;f=a.o+a.u;var g=e.fa;g>d&&(g=d);0===g?c=0:(e.fa-=g,L.rb(c,e.input,e.jb,g,f),1===e.state.wrap?e.C=yj(e.C,c,g,f):2===e.state.wrap&&(e.C=zj(e.C,c,g,f)),e.jb+=g,e.mb+=g,c=g);a.u+=c;if(3<=a.u+a.la)for(d=a.o-a.la,a.G=a.window[d],
a.G=(a.G<<a.Ia^a.window[d+1])&a.Ha;a.la&&!(a.G=(a.G<<a.Ia^a.window[d+3-1])&a.Ha,a.Da[d&a.Wa]=a.head[a.G],a.head[a.G]=d,d++,a.la--,3>a.u+a.la););}while(262>a.u&&0!==a.D.fa)}
function tk(a,b){for(var c;;){if(262>a.u){sk(a);if(262>a.u&&0===b)return 1;if(0===a.u)break}c=0;3<=a.u&&(a.G=(a.G<<a.Ia^a.window[a.o+3-1])&a.Ha,c=a.Da[a.o&a.Wa]=a.head[a.G],a.head[a.G]=a.o);0!==c&&a.o-c<=a.da-262&&(a.J=rk(a,c));if(3<=a.J)if(c=lk(a,a.o-a.xb,a.J-3),a.u-=a.J,a.J<=a.Hc&&3<=a.u){a.J--;do a.o++,a.G=(a.G<<a.Ia^a.window[a.o+3-1])&a.Ha,a.Da[a.o&a.Wa]=a.head[a.G],a.head[a.G]=a.o;while(0!==--a.J);a.o++}else a.o+=a.J,a.J=0,a.G=a.window[a.o],a.G=(a.G<<a.Ia^a.window[a.o+1])&a.Ha;else c=lk(a,0,
a.window[a.o]),a.u--,a.o++;if(c&&(pk(a,!1),0===a.D.I))return 1}a.la=2>a.o?a.o:2;return 4===b?(pk(a,!0),0===a.D.I?3:4):a.sa&&(pk(a,!1),0===a.D.I)?1:2}
function uk(a,b){for(var c,d;;){if(262>a.u){sk(a);if(262>a.u&&0===b)return 1;if(0===a.u)break}c=0;3<=a.u&&(a.G=(a.G<<a.Ia^a.window[a.o+3-1])&a.Ha,c=a.Da[a.o&a.Wa]=a.head[a.G],a.head[a.G]=a.o);a.qa=a.J;a.Dd=a.xb;a.J=2;0!==c&&a.qa<a.Hc&&a.o-c<=a.da-262&&(a.J=rk(a,c),5>=a.J&&(1===a.strategy||3===a.J&&4096<a.o-a.xb)&&(a.J=2));if(3<=a.qa&&a.J<=a.qa){d=a.o+a.u-3;c=lk(a,a.o-1-a.Dd,a.qa-3);a.u-=a.qa-1;a.qa-=2;do++a.o<=d&&(a.G=(a.G<<a.Ia^a.window[a.o+3-1])&a.Ha,a.Da[a.o&a.Wa]=a.head[a.G],a.head[a.G]=a.o);
while(0!==--a.qa);a.hb=0;a.J=2;a.o++;if(c&&(pk(a,!1),0===a.D.I))return 1}else if(a.hb){if((c=lk(a,0,a.window[a.o-1]))&&pk(a,!1),a.o++,a.u--,0===a.D.I)return 1}else a.hb=1,a.o++,a.u--}a.hb&&(lk(a,0,a.window[a.o-1]),a.hb=0);a.la=2>a.o?a.o:2;return 4===b?(pk(a,!0),0===a.D.I?3:4):a.sa&&(pk(a,!1),0===a.D.I)?1:2}
function vk(a,b){for(var c,d,e,f=a.window;;){if(258>=a.u){sk(a);if(258>=a.u&&0===b)return 1;if(0===a.u)break}a.J=0;if(3<=a.u&&0<a.o&&(d=a.o-1,c=f[d],c===f[++d]&&c===f[++d]&&c===f[++d])){for(e=a.o+258;c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&c===f[++d]&&d<e;);a.J=258-(e-d);a.J>a.u&&(a.J=a.u)}3<=a.J?(c=lk(a,1,a.J-3),a.u-=a.J,a.o+=a.J,a.J=0):(c=lk(a,0,a.window[a.o]),a.u--,a.o++);if(c&&(pk(a,!1),0===a.D.I))return 1}a.la=0;return 4===b?(pk(a,!0),0===a.D.I?3:4):
a.sa&&(pk(a,!1),0===a.D.I)?1:2}
function wk(a,b){for(var c;;){if(0===a.u&&(sk(a),0===a.u)){if(0===b)return 1;break}a.J=0;c=lk(a,0,a.window[a.o]);a.u--;a.o++;if(c&&(pk(a,!1),0===a.D.I))return 1}a.la=0;return 4===b?(pk(a,!0),0===a.D.I?3:4):a.sa&&(pk(a,!1),0===a.D.I)?1:2}
function xk(a,b,c,d,e){this.we=a;this.Fe=b;this.Je=c;this.Ee=d;this.re=e}
var yk;yk=[new xk(0,0,0,0,function(a,b){var c=65535;for(c>a.ta-5&&(c=a.ta-5);;){if(1>=a.u){sk(a);if(0===a.u&&0===b)return 1;if(0===a.u)break}a.o+=a.u;a.u=0;var d=a.na+c;if(0===a.o||a.o>=d)if(a.u=a.o-d,a.o=d,pk(a,!1),0===a.D.I)return 1;if(a.o-a.na>=a.da-262&&(pk(a,!1),0===a.D.I))return 1}a.la=0;if(4===b)return pk(a,!0),0===a.D.I?3:4;a.o>a.na&&pk(a,!1);return 1}),
new xk(4,4,8,4,tk),new xk(4,5,16,8,tk),new xk(4,6,32,32,tk),new xk(4,4,16,16,uk),new xk(8,16,32,32,uk),new xk(8,16,128,128,uk),new xk(8,32,128,256,uk),new xk(32,128,258,1024,uk),new xk(32,258,258,4096,uk)];
function zk(){this.D=null;this.status=0;this.R=null;this.wrap=this.pending=this.Ob=this.ta=0;this.B=null;this.wa=0;this.method=8;this.wb=-1;this.Wa=this.Uc=this.da=0;this.window=null;this.Td=0;this.head=this.Da=null;this.Ad=this.rd=this.strategy=this.level=this.Hc=this.yd=this.qa=this.u=this.xb=this.o=this.hb=this.Dd=this.J=this.na=this.Ia=this.Ha=this.Cc=this.ic=this.G=0;this.ka=new L.Fa(1146);this.gb=new L.Fa(122);this.ba=new L.Fa(78);nk(this.ka);nk(this.gb);nk(this.ba);this.dd=this.ec=this.jc=
null;this.Ga=new L.Fa(16);this.S=new L.Fa(573);nk(this.S);this.vb=this.Ja=0;this.depth=new L.Fa(573);nk(this.depth);this.Z=this.ga=this.la=this.matches=this.Bb=this.Ma=this.Hb=this.sa=this.Mb=this.Gc=0}
function Ak(a,b){if(!a||!a.state||5<b||0>b)return a?mk(a,-2):-2;var c=a.state;if(!a.Nb||!a.input&&0!==a.fa||666===c.status&&4!==b)return mk(a,0===a.I?-5:-2);c.D=a;var d=c.wb;c.wb=b;if(42===c.status)if(2===c.wrap)a.C=0,N(c,31),N(c,139),N(c,8),c.B?(N(c,(c.B.text?1:0)+(c.B.Ra?2:0)+(c.B.Qa?4:0)+(c.B.name?8:0)+(c.B.comment?16:0)),N(c,c.B.time&255),N(c,c.B.time>>8&255),N(c,c.B.time>>16&255),N(c,c.B.time>>24&255),N(c,9===c.level?2:2<=c.strategy||2>c.level?4:0),N(c,c.B.os&255),c.B.Qa&&c.B.Qa.length&&(N(c,
c.B.Qa.length&255),N(c,c.B.Qa.length>>8&255)),c.B.Ra&&(a.C=zj(a.C,c.R,c.pending,0)),c.wa=0,c.status=69):(N(c,0),N(c,0),N(c,0),N(c,0),N(c,0),N(c,9===c.level?2:2<=c.strategy||2>c.level?4:0),N(c,3),c.status=113);else{var e=8+(c.Uc-8<<4)<<8;e|=(2<=c.strategy||2>c.level?0:6>c.level?1:6===c.level?2:3)<<6;0!==c.o&&(e|=32);c.status=113;qk(c,e+(31-e%31));0!==c.o&&(qk(c,a.C>>>16),qk(c,a.C&65535));a.C=1}if(69===c.status)if(c.B.Qa){for(e=c.pending;c.wa<(c.B.Qa.length&65535)&&(c.pending!==c.ta||(c.B.Ra&&c.pending>
e&&(a.C=zj(a.C,c.R,c.pending-e,e)),ok(a),e=c.pending,c.pending!==c.ta));)N(c,c.B.Qa[c.wa]&255),c.wa++;c.B.Ra&&c.pending>e&&(a.C=zj(a.C,c.R,c.pending-e,e));c.wa===c.B.Qa.length&&(c.wa=0,c.status=73)}else c.status=73;if(73===c.status)if(c.B.name){e=c.pending;do{if(c.pending===c.ta&&(c.B.Ra&&c.pending>e&&(a.C=zj(a.C,c.R,c.pending-e,e)),ok(a),e=c.pending,c.pending===c.ta)){var f=1;break}f=c.wa<c.B.name.length?c.B.name.charCodeAt(c.wa++)&255:0;N(c,f)}while(0!==f);c.B.Ra&&c.pending>e&&(a.C=zj(a.C,c.R,c.pending-
e,e));0===f&&(c.wa=0,c.status=91)}else c.status=91;if(91===c.status)if(c.B.comment){e=c.pending;do{if(c.pending===c.ta&&(c.B.Ra&&c.pending>e&&(a.C=zj(a.C,c.R,c.pending-e,e)),ok(a),e=c.pending,c.pending===c.ta)){f=1;break}f=c.wa<c.B.comment.length?c.B.comment.charCodeAt(c.wa++)&255:0;N(c,f)}while(0!==f);c.B.Ra&&c.pending>e&&(a.C=zj(a.C,c.R,c.pending-e,e));0===f&&(c.status=103)}else c.status=103;103===c.status&&(c.B.Ra?(c.pending+2>c.ta&&ok(a),c.pending+2<=c.ta&&(N(c,a.C&255),N(c,a.C>>8&255),a.C=0,
c.status=113)):c.status=113);if(0!==c.pending){if(ok(a),0===a.I)return c.wb=-1,0}else if(0===a.fa&&(b<<1)-(4<b?9:0)<=(d<<1)-(4<d?9:0)&&4!==b)return mk(a,-5);if(666===c.status&&0!==a.fa)return mk(a,-5);if(0!==a.fa||0!==c.u||0!==b&&666!==c.status){d=2===c.strategy?wk(c,b):3===c.strategy?vk(c,b):yk[c.level].re(c,b);if(3===d||4===d)c.status=666;if(1===d||3===d)return 0===a.I&&(c.wb=-1),0;if(2===d&&(1===b?(Wj(c,2,3),Xj(c,256,Kj),16===c.Z?(Vj(c,c.ga),c.ga=0,c.Z=0):8<=c.Z&&(c.R[c.pending++]=c.ga&255,c.ga>>=
8,c.Z-=8)):5!==b&&(Wj(c,0,3),ck(c,0,0),3===b&&(nk(c.head),0===c.u&&(c.o=0,c.na=0,c.la=0))),ok(a),0===a.I))return c.wb=-1,0}if(4!==b)return 0;if(0>=c.wrap)return 1;2===c.wrap?(N(c,a.C&255),N(c,a.C>>8&255),N(c,a.C>>16&255),N(c,a.C>>24&255),N(c,a.mb&255),N(c,a.mb>>8&255),N(c,a.mb>>16&255),N(c,a.mb>>24&255)):(qk(c,a.C>>>16),qk(c,a.C&65535));ok(a);0<c.wrap&&(c.wrap=-c.wrap);return 0!==c.pending?0:1}
;var Bk={};Bk=function(){this.input=null;this.mb=this.fa=this.jb=0;this.Nb=null;this.Sc=this.I=this.zb=0;this.msg="";this.state=null;this.wc=2;this.C=0};var Ck=Object.prototype.toString;
function Dk(a){if(!(this instanceof Dk))return new Dk(a);a=this.options=L.assign({level:-1,method:8,chunkSize:16384,Xa:15,He:8,strategy:0,K:""},a||{});a.raw&&0<a.Xa?a.Xa=-a.Xa:a.xe&&0<a.Xa&&16>a.Xa&&(a.Xa+=16);this.err=0;this.msg="";this.ended=!1;this.chunks=[];this.D=new Bk;this.D.I=0;var b=this.D;var c=a.level,d=a.method,e=a.Xa,f=a.He,g=a.strategy;if(b){var h=1;-1===c&&(c=6);0>e?(h=0,e=-e):15<e&&(h=2,e-=16);if(1>f||9<f||8!==d||8>e||15<e||0>c||9<c||0>g||4<g)b=mk(b,-2);else{8===e&&(e=9);var l=new zk;
b.state=l;l.D=b;l.wrap=h;l.B=null;l.Uc=e;l.da=1<<l.Uc;l.Wa=l.da-1;l.Cc=f+7;l.ic=1<<l.Cc;l.Ha=l.ic-1;l.Ia=~~((l.Cc+3-1)/3);l.window=new L.bb(2*l.da);l.head=new L.Fa(l.ic);l.Da=new L.Fa(l.da);l.Mb=1<<f+6;l.ta=4*l.Mb;l.R=new L.bb(l.ta);l.Hb=1*l.Mb;l.Gc=3*l.Mb;l.level=c;l.strategy=g;l.method=d;if(b&&b.state){b.mb=b.Sc=0;b.wc=2;c=b.state;c.pending=0;c.Ob=0;0>c.wrap&&(c.wrap=-c.wrap);c.status=c.wrap?42:113;b.C=2===c.wrap?0:1;c.wb=0;if(!kk){d=Array(16);for(f=g=0;28>f;f++)for(Oj[f]=g,e=0;e<1<<Gj[f];e++)Nj[g++]=
f;Nj[g-1]=f;for(f=g=0;16>f;f++)for(Pj[f]=g,e=0;e<1<<Hj[f];e++)Mj[g++]=f;for(g>>=7;30>f;f++)for(Pj[f]=g<<7,e=0;e<1<<Hj[f]-7;e++)Mj[256+g++]=f;for(e=0;15>=e;e++)d[e]=0;for(e=0;143>=e;)Kj[2*e+1]=8,e++,d[8]++;for(;255>=e;)Kj[2*e+1]=9,e++,d[9]++;for(;279>=e;)Kj[2*e+1]=7,e++,d[7]++;for(;287>=e;)Kj[2*e+1]=8,e++,d[8]++;Zj(Kj,287,d);for(e=0;30>e;e++)Lj[2*e+1]=5,Lj[2*e]=Yj(e,5);Rj=new Qj(Kj,Gj,257,286,15);Sj=new Qj(Lj,Hj,0,30,15);Tj=new Qj([],Ij,0,19,7);kk=!0}c.jc=new Uj(c.ka,Rj);c.ec=new Uj(c.gb,Sj);c.dd=
new Uj(c.ba,Tj);c.ga=0;c.Z=0;ak(c);c=0}else c=mk(b,-2);0===c&&(b=b.state,b.Td=2*b.da,nk(b.head),b.Hc=yk[b.level].Fe,b.rd=yk[b.level].we,b.Ad=yk[b.level].Je,b.yd=yk[b.level].Ee,b.o=0,b.na=0,b.u=0,b.la=0,b.J=b.qa=2,b.hb=0,b.G=0);b=c}}else b=-2;if(0!==b)throw Error(Ej[b]);a.header&&(b=this.D)&&b.state&&2===b.state.wrap&&(b.state.B=a.header);if(a.Ib){var m;"string"===typeof a.Ib?m=xj(a.Ib):"[object ArrayBuffer]"===Ck.call(a.Ib)?m=new Uint8Array(a.Ib):m=a.Ib;a=this.D;f=m;g=f.length;if(a&&a.state)if(m=
a.state,b=m.wrap,2===b||1===b&&42!==m.status||m.u)b=-2;else{1===b&&(a.C=yj(a.C,f,g,0));m.wrap=0;g>=m.da&&(0===b&&(nk(m.head),m.o=0,m.na=0,m.la=0),c=new L.bb(m.da),L.rb(c,f,g-m.da,m.da,0),f=c,g=m.da);c=a.fa;d=a.jb;e=a.input;a.fa=g;a.jb=0;a.input=f;for(sk(m);3<=m.u;){f=m.o;g=m.u-2;do m.G=(m.G<<m.Ia^m.window[f+3-1])&m.Ha,m.Da[f&m.Wa]=m.head[m.G],m.head[m.G]=f,f++;while(--g);m.o=f;m.u=2;sk(m)}m.o+=m.u;m.na=m.o;m.la=m.u;m.u=0;m.J=m.qa=2;m.hb=0;a.jb=d;a.input=e;a.fa=c;m.wrap=b;b=0}else b=-2;if(0!==b)throw Error(Ej[b]);
this.mg=!0}}
Dk.prototype.push=function(a,b){var c=this.D,d=this.options.chunkSize;if(this.ended)return!1;var e=b===~~b?b:!0===b?4:0;"string"===typeof a?c.input=xj(a):"[object ArrayBuffer]"===Ck.call(a)?c.input=new Uint8Array(a):c.input=a;c.jb=0;c.fa=c.input.length;do{0===c.I&&(c.Nb=new L.bb(d),c.zb=0,c.I=d);a=Ak(c,e);if(1!==a&&0!==a)return Ek(this,a),this.ended=!0,!1;if(0===c.I||0===c.fa&&(4===e||2===e))if("string"===this.options.K){var f=L.Rc(c.Nb,c.zb);b=f;f=f.length;if(65537>f&&(b.subarray&&uj||!b.subarray))b=
String.fromCharCode.apply(null,L.Rc(b,f));else{for(var g="",h=0;h<f;h++)g+=String.fromCharCode(b[h]);b=g}this.chunks.push(b)}else b=L.Rc(c.Nb,c.zb),this.chunks.push(b)}while((0<c.fa||0===c.I)&&1!==a);if(4===e)return(c=this.D)&&c.state?(d=c.state.status,42!==d&&69!==d&&73!==d&&91!==d&&103!==d&&113!==d&&666!==d?a=mk(c,-2):(c.state=null,a=113===d?mk(c,-3):0)):a=-2,Ek(this,a),this.ended=!0,0===a;2===e&&(Ek(this,0),c.I=0);return!0};
function Ek(a,b){0===b&&(a.result="string"===a.options.K?a.chunks.join(""):L.od(a.chunks));a.chunks=[];a.err=b;a.msg=a.D.msg}
function Fk(a){var b=b||{};b.xe=!0;b=new Dk(b);b.push(a,!0);if(b.err)throw b.msg||Ej[b.err];return b.result}
;function Gk(a){return Hb(null===a?"null":void 0===a?"undefined":a)}
;function Hk(a){this.name=a}
;var Ik=new Hk("rawColdConfigGroup");var Jk=new Hk("rawHotConfigGroup");function Kk(a){I.call(this,a)}
v(Kk,I);function Lk(a,b){return me(a,1,b)}
;function Mk(a){I.call(this,a,-1,Nk)}
v(Mk,I);var Nk=[1];function Ok(a){I.call(this,a)}
v(Ok,I);function Pk(a){I.call(this,a)}
v(Pk,I);function Qk(a){I.call(this,a)}
v(Qk,I);function Rk(a){I.call(this,a,-1,Sk)}
v(Rk,I);var Sk=[2];function Tk(a){I.call(this,a,-1,Uk)}
v(Tk,I);Tk.prototype.getPlayerType=function(){return Zd(this,36)};
Tk.prototype.setHomeGroupInfo=function(a){return H(this,Rk,81,a)};
Tk.prototype.clearLocationPlayabilityToken=function(){return G(this,89,void 0,!1)};
var Uk=[9,66,24,32,86,100,101];function Vk(a){I.call(this,a)}
v(Vk,I);Vk.prototype.getKey=function(){return qe(this,1)};
Vk.prototype.getValue=function(){return qe(this,2===fe(this,Wk)?2:-1)};
var Wk=[2,3,4,5,6];function Xk(a){I.call(this,a,-1,Yk)}
v(Xk,I);var Yk=[15,26,28];function Zk(a){I.call(this,a,-1,$k)}
v(Zk,I);var $k=[5];function al(a){I.call(this,a)}
v(al,I);function bl(a){I.call(this,a,-1,cl)}
v(bl,I);bl.prototype.setSafetyMode=function(a){return G(this,5,a)};
var cl=[12];function dl(a){I.call(this,a,-1,el)}
v(dl,I);dl.prototype.l=function(a){return H(this,Tk,1,a)};
var el=[12];var fl=new Hk("continuationCommand");var gl=new Hk("webCommandMetadata");var hl=new Hk("signalServiceEndpoint");var il={Hf:"EMBEDDED_PLAYER_MODE_UNKNOWN",Ef:"EMBEDDED_PLAYER_MODE_DEFAULT",Gf:"EMBEDDED_PLAYER_MODE_PFP",Ff:"EMBEDDED_PLAYER_MODE_PFL"};var jl=new Hk("feedbackEndpoint");var Bl={lg:"WEB_DISPLAY_MODE_UNKNOWN",hg:"WEB_DISPLAY_MODE_BROWSER",jg:"WEB_DISPLAY_MODE_MINIMAL_UI",kg:"WEB_DISPLAY_MODE_STANDALONE",ig:"WEB_DISPLAY_MODE_FULLSCREEN"};function Cl(a){I.call(this,a)}
v(Cl,I);Cl.prototype.getKey=function(){return qe(this,1)};
Cl.prototype.getValue=function(){return qe(this,2)};function Dl(a){I.call(this,a,-1,El)}
v(Dl,I);var El=[4,5];function Fl(a){I.call(this,a)}
v(Fl,I);function Gl(a){I.call(this,a)}
v(Gl,I);var Hl=[2,3,4];function Il(a){I.call(this,a)}
v(Il,I);Il.prototype.getMessage=function(){return qe(this,1)};function Jl(a){I.call(this,a)}
v(Jl,I);function Kl(a){I.call(this,a)}
v(Kl,I);function Ll(a){I.call(this,a,-1,Ml)}
v(Ll,I);var Ml=[10,17];function Nl(a){I.call(this,a)}
v(Nl,I);function Ol(a){I.call(this,a)}
v(Ol,I);function Pl(a){I.call(this,a)}
v(Pl,I);function Ql(a){I.call(this,a)}
v(Ql,I);function Rl(a){I.call(this,a)}
v(Rl,I);function Sl(a){I.call(this,a,-1,Tl)}
v(Sl,I);Sl.prototype.getVideoData=function(){return ge(this,Rl,15)};
var Tl=[4];function Ul(a){I.call(this,a)}
v(Ul,I);function Vl(a,b){H(a,Pl,1,b)}
;function Wl(a){I.call(this,a)}
v(Wl,I);function Xl(a,b){return H(a,Pl,1,b)}
Wl.prototype.h=function(a){return G(this,2,a)};function Yl(a){I.call(this,a,-1,Zl)}
v(Yl,I);Yl.prototype.h=function(a){return G(this,1,a)};
function $l(a,b){return H(a,Pl,2,b)}
var Zl=[3];function am(a){I.call(this,a)}
v(am,I);am.prototype.h=function(a){return G(this,1,a)};function bm(a){I.call(this,a)}
v(bm,I);bm.prototype.h=function(a){return G(this,1,a)};function cm(a){I.call(this,a)}
v(cm,I);cm.prototype.h=function(a){return G(this,1,a)};function dm(a){I.call(this,a)}
v(dm,I);dm.prototype.h=function(a){return G(this,1,a)};function em(a){I.call(this,a)}
v(em,I);function fm(a){I.call(this,a)}
v(fm,I);function gm(a){var b=new fm;return G(b,1,a)}
fm.prototype.getId=function(){return qe(this,2)};
function hm(a,b){return G(a,2,b)}
;function im(a){I.call(this,a)}
v(im,I);function jm(a){I.call(this,a,-1,km)}
v(jm,I);jm.prototype.getPlayerType=function(){return pe(Zd(this,7),0)};
jm.prototype.setVideoId=function(a){return G(this,19,a)};
function lm(a,b){le(a,68,fm,b)}
var km=[83,68];function mm(a){I.call(this,a)}
v(mm,I);function nm(a){I.call(this,a)}
v(nm,I);function om(a){I.call(this,a)}
v(om,I);function pm(a){I.call(this,a,470)}
v(pm,I);
var qm=[2,3,5,6,7,11,13,20,21,22,23,24,28,32,37,45,59,72,73,74,76,78,79,80,85,91,97,100,102,105,111,117,119,126,127,136,146,148,151,156,157,158,159,163,164,168,176,177,178,179,184,188,189,190,191,193,194,195,196,197,198,199,200,201,202,203,204,205,206,208,209,215,219,222,225,226,227,229,232,233,234,240,241,244,247,248,249,251,254,255,256,257,258,259,260,261,266,270,272,278,288,291,293,300,304,308,309,310,311,313,314,319,320,321,323,324,327,328,330,331,332,334,337,338,340,344,348,350,351,352,353,354,
355,356,357,358,361,363,364,368,369,370,373,374,375,378,380,381,383,388,389,402,403,410,411,412,413,414,415,416,417,418,423,424,425,426,427,429,430,431,439,441,444,448,458,469];var rm={Zf:0,If:1,Of:2,Pf:4,Vf:8,Qf:16,Rf:32,Yf:64,Xf:128,Kf:256,Mf:512,Tf:1024,Lf:2048,Nf:4096,Jf:8192,Sf:16384,Wf:32768,Uf:65536};function sm(a){I.call(this,a)}
v(sm,I);function tm(a){I.call(this,a)}
v(tm,I);tm.prototype.setVideoId=function(a){return ee(this,1,um,a)};
tm.prototype.getPlaylistId=function(){var a=2===fe(this,um)?2:-1;return Zd(this,a)};
var um=[1,2];function vm(a){I.call(this,a,-1,wm)}
v(vm,I);var wm=[3];var xm=new Hk("webPlayerShareEntityServiceEndpoint");var ym=new Hk("playlistEditEndpoint");var zm=new Hk("modifyChannelNotificationPreferenceEndpoint");var Am=new Hk("unsubscribeEndpoint");var Bm=new Hk("subscribeEndpoint");function Cm(){var a=Dm;B("yt.ads.biscotti.getId_")||z("yt.ads.biscotti.getId_",a)}
function Em(a){z("yt.ads.biscotti.lastId_",a)}
;function Fm(a,b){1<b.length?a[b[0]]=b[1]:1===b.length&&Object.assign(a,b[0])}
;var Gm=y.window,Hm,Im,Jm=(null==Gm?void 0:null==(Hm=Gm.yt)?void 0:Hm.config_)||(null==Gm?void 0:null==(Im=Gm.ytcfg)?void 0:Im.data_)||{};z("yt.config_",Jm);function Km(){Fm(Jm,arguments)}
function O(a,b){return a in Jm?Jm[a]:b}
function Lm(){var a=Jm.EXPERIMENT_FLAGS;return a?a.web_disable_gel_stp_ecatcher_killswitch:void 0}
;var Mm=[];function Nm(a){Mm.forEach(function(b){return b(a)})}
function Om(a){return a&&window.yterr?function(){try{return a.apply(this,arguments)}catch(b){Pm(b)}}:a}
function Pm(a,b,c,d,e){var f=B("yt.logging.errors.log");f?f(a,"ERROR",b,c,d,void 0,e):(f=O("ERRORS",[]),f.push([a,"ERROR",b,c,d,void 0,e]),Km("ERRORS",f));Nm(a)}
function Qm(a,b,c,d,e){var f=B("yt.logging.errors.log");f?f(a,"WARNING",b,c,d,void 0,e):(f=O("ERRORS",[]),f.push([a,"WARNING",b,c,d,void 0,e]),Km("ERRORS",f))}
;var Rm=/^[\w.]*$/,Sm={q:!0,search_query:!0};function Tm(a,b){b=a.split(b);for(var c={},d=0,e=b.length;d<e;d++){var f=b[d].split("=");if(1==f.length&&f[0]||2==f.length)try{var g=Um(f[0]||""),h=Um(f[1]||"");g in c?Array.isArray(c[g])?lb(c[g],h):c[g]=[c[g],h]:c[g]=h}catch(q){var l=q,m=f[0],n=String(Tm);l.args=[{key:m,value:f[1],query:a,method:Vm==n?"unchanged":n}];Sm.hasOwnProperty(m)||Qm(l)}}return c}
var Vm=String(Tm);function Wm(a){var b=[];mb(a,function(c,d){var e=encodeURIComponent(String(d)),f;Array.isArray(c)?f=c:f=[c];fb(f,function(g){""==g?b.push(e):b.push(e+"="+encodeURIComponent(String(g)))})});
return b.join("&")}
function Xm(a){"?"==a.charAt(0)&&(a=a.substr(1));return Tm(a,"&")}
function Ym(a){return-1!=a.indexOf("?")?(a=(a||"").split("#")[0],a=a.split("?",2),Xm(1<a.length?a[1]:a[0])):{}}
function Zm(a,b,c){var d=a.split("#",2);a=d[0];d=1<d.length?"#"+d[1]:"";var e=a.split("?",2);a=e[0];e=Xm(e[1]||"");for(var f in b)!c&&null!==e&&f in e||(e[f]=b[f]);return Dc(a,e)+d}
function $m(a){if(!b)var b=window.location.href;var c=vc(1,a),d=wc(a);c&&d?(a=a.match(tc),b=b.match(tc),a=a[3]==b[3]&&a[1]==b[1]&&a[4]==b[4]):a=d?wc(b)==d&&(Number(vc(4,b))||null)==(Number(vc(4,a))||null):!0;return a}
function Um(a){return a&&a.match(Rm)?a:decodeURIComponent(a.replace(/\+/g," "))}
;function an(a){var b=bn;a=void 0===a?B("yt.ads.biscotti.lastId_")||"":a;var c=Object,d=c.assign,e={};e.dt=vi;e.flash="0";a:{try{var f=b.h.top.location.href}catch(J){f=2;break a}f=f?f===b.i.location.href?0:1:2}e=(e.frm=f,e);try{e.u_tz=-(new Date).getTimezoneOffset();var g=void 0===g?bi:g;try{var h=g.history.length}catch(J){h=0}e.u_his=h;var l;e.u_h=null==(l=bi.screen)?void 0:l.height;var m;e.u_w=null==(m=bi.screen)?void 0:m.width;var n;e.u_ah=null==(n=bi.screen)?void 0:n.availHeight;var q;e.u_aw=null==
(q=bi.screen)?void 0:q.availWidth;var u;e.u_cd=null==(u=bi.screen)?void 0:u.colorDepth}catch(J){}h=b.h;try{var t=h.screenX;var A=h.screenY}catch(J){}try{var C=h.outerWidth;var E=h.outerHeight}catch(J){}try{var M=h.innerWidth;var Q=h.innerHeight}catch(J){}try{var S=h.screenLeft;var V=h.screenTop}catch(J){}try{M=h.innerWidth,Q=h.innerHeight}catch(J){}try{var Z=h.screen.availWidth;var bb=h.screen.availTop}catch(J){}t=[S,V,t,A,Z,bb,C,E,M,Q];try{var Qa=(b.h.top||window).document,Da="CSS1Compat"==Qa.compatMode?
Qa.documentElement:Qa.body;var sa=(new Af(Da.clientWidth,Da.clientHeight)).round()}catch(J){sa=new Af(-12245933,-12245933)}Qa=sa;sa={};var ia=void 0===ia?y:ia;Da=new Ei;ia.SVGElement&&ia.document.createElementNS&&Da.set(0);A=si();A["allow-top-navigation-by-user-activation"]&&Da.set(1);A["allow-popups-to-escape-sandbox"]&&Da.set(2);ia.crypto&&ia.crypto.subtle&&Da.set(3);ia.TextDecoder&&ia.TextEncoder&&Da.set(4);ia=Fi(Da);sa.bc=ia;sa.bih=Qa.height;sa.biw=Qa.width;sa.brdim=t.join();b=b.i;b=(sa.vis=b.prerendering?
3:{visible:1,hidden:2,prerender:3,preview:4,unloaded:5}[b.visibilityState||b.webkitVisibilityState||b.mozVisibilityState||""]||0,sa.wgl=!!bi.WebGLRenderingContext,sa);c=d.call(c,e,b);c.ca_type="image";a&&(c.bid=a);return c}
var bn=new function(){var a=window.document;this.h=window;this.i=a};
z("yt.ads_.signals_.getAdSignalsString",function(a){return Wm(an(a))});Date.now();function P(a){a=cn(a);return"string"===typeof a&&"false"===a?!1:!!a}
function dn(a,b){a=cn(a);return void 0===a&&void 0!==b?b:Number(a||0)}
function en(){return O("EXPERIMENTS_TOKEN","")}
function cn(a){var b=O("EXPERIMENTS_FORCED_FLAGS",{})||{};return void 0!==b[a]?b[a]:O("EXPERIMENT_FLAGS",{})[a]}
function fn(){for(var a=[],b=O("EXPERIMENTS_FORCED_FLAGS",{}),c=r(Object.keys(b)),d=c.next();!d.done;d=c.next())d=d.value,a.push({key:d,value:String(b[d])});c=O("EXPERIMENT_FLAGS",{});var e=r(Object.keys(c));for(d=e.next();!d.done;d=e.next())d=d.value,d.startsWith("force_")&&void 0===b[d]&&a.push({key:d,value:String(c[d])});return a}
;var gn="XMLHttpRequest"in y?function(){return new XMLHttpRequest}:null;
function hn(){if(!gn)return null;var a=gn();return"open"in a?a:null}
function jn(a){switch(a&&"status"in a?a.status:-1){case 200:case 201:case 202:case 203:case 204:case 205:case 206:case 304:return!0;default:return!1}}
;function kn(a,b){"function"===typeof a&&(a=Om(a));return window.setTimeout(a,b)}
;var ln={Authorization:"AUTHORIZATION","X-Goog-EOM-Visitor-Id":"EOM_VISITOR_DATA","X-Goog-Visitor-Id":"SANDBOXED_VISITOR_ID","X-Youtube-Domain-Admin-State":"DOMAIN_ADMIN_STATE","X-Youtube-Chrome-Connected":"CHROME_CONNECTED_HEADER","X-YouTube-Client-Name":"INNERTUBE_CONTEXT_CLIENT_NAME","X-YouTube-Client-Version":"INNERTUBE_CONTEXT_CLIENT_VERSION","X-YouTube-Delegation-Context":"INNERTUBE_CONTEXT_SERIALIZED_DELEGATION_CONTEXT","X-YouTube-Device":"DEVICE","X-Youtube-Identity-Token":"ID_TOKEN","X-YouTube-Page-CL":"PAGE_CL",
"X-YouTube-Page-Label":"PAGE_BUILD_LABEL","X-YouTube-Variants-Checksum":"VARIANTS_CHECKSUM","X-Goog-AuthUser":"SESSION_INDEX","X-Goog-PageId":"DELEGATED_SESSION_ID"},mn="app debugcss debugjs expflag force_ad_params force_ad_encrypted force_viral_ad_response_params forced_experiments innertube_snapshots innertube_goldens internalcountrycode internalipoverride absolute_experiments conditional_experiments sbb sr_bns_address".split(" ").concat(ha(wi)),nn=!1;
function on(a,b){b=void 0===b?{}:b;var c=$m(a),d=P("web_ajax_ignore_global_headers_if_set"),e;for(e in ln){var f=O(ln[e]);"X-Goog-Visitor-Id"!==e||f||(f=O("VISITOR_DATA"));!f||!c&&wc(a)||d&&void 0!==b[e]||!(P("move_vss_away_from_login_info_cookie")||"X-Goog-AuthUser"!==e&&"X-Goog-PageId"!==e)||(b[e]=f)}P("move_vss_away_from_login_info_cookie")&&(b["X-Yt-Auth-Test"]="test");"X-Goog-EOM-Visitor-Id"in b&&"X-Goog-Visitor-Id"in b&&delete b["X-Goog-Visitor-Id"];if(c||!wc(a))b["X-YouTube-Utc-Offset"]=String(-(new Date).getTimezoneOffset());
if(c||!wc(a)){try{var g=(new Intl.DateTimeFormat).resolvedOptions().timeZone}catch(h){}g&&(b["X-YouTube-Time-Zone"]=g)}document.location.hostname.endsWith("youtubeeducation.com")||!c&&wc(a)||(b["X-YouTube-Ad-Signals"]=Wm(an()));return b}
function pn(a){var b=window.location.search,c=wc(a);P("debug_handle_relative_url_for_query_forward_killswitch")||!c&&$m(a)&&(c=document.location.hostname);var d=uc(vc(5,a));d=(c=c&&(c.endsWith("youtube.com")||c.endsWith("youtube-nocookie.com")))&&d&&d.startsWith("/api/");if(!c||d)return a;var e=Xm(b),f={};fb(mn,function(g){e[g]&&(f[g]=e[g])});
return Zm(a,f||{},!1)}
function qn(a,b){var c=b.format||"JSON";a=rn(a,b);var d=sn(a,b),e=!1,f=tn(a,function(l){if(!e){e=!0;h&&window.clearTimeout(h);var m=jn(l),n=null,q=400<=l.status&&500>l.status,u=500<=l.status&&600>l.status;if(m||q||u)n=un(a,c,l,b.convertToSafeHtml);if(m)a:if(l&&204==l.status)m=!0;else{switch(c){case "XML":m=0==parseInt(n&&n.return_code,10);break a;case "RAW":m=!0;break a}m=!!n}n=n||{};q=b.context||y;m?b.onSuccess&&b.onSuccess.call(q,l,n):b.onError&&b.onError.call(q,l,n);b.onFinish&&b.onFinish.call(q,
l,n)}},b.method,d,b.headers,b.responseType,b.withCredentials);
d=b.timeout||0;if(b.onTimeout&&0<d){var g=b.onTimeout;var h=kn(function(){e||(e=!0,f.abort(),window.clearTimeout(h),g.call(b.context||y,f))},d)}return f}
function rn(a,b){b.includeDomain&&(a=document.location.protocol+"//"+document.location.hostname+(document.location.port?":"+document.location.port:"")+a);var c=O("XSRF_FIELD_NAME");if(b=b.urlParams)b[c]&&delete b[c],a=Zm(a,b||{},!0);return a}
function sn(a,b){var c=O("XSRF_FIELD_NAME"),d=O("XSRF_TOKEN"),e=b.postBody||"",f=b.postParams,g=O("XSRF_FIELD_NAME"),h;b.headers&&(h=b.headers["Content-Type"]);b.excludeXsrf||wc(a)&&!b.withCredentials&&wc(a)!=document.location.hostname||"POST"!=b.method||h&&"application/x-www-form-urlencoded"!=h||b.postParams&&b.postParams[g]||(f||(f={}),f[c]=d);(P("ajax_parse_query_data_only_when_filled")&&f&&0<Object.keys(f).length||f)&&"string"===typeof e&&(e=Xm(e),wb(e,f),e=b.postBodyFormat&&"JSON"==b.postBodyFormat?
JSON.stringify(e):Bc(e));f=e||f&&!pb(f);!nn&&f&&"POST"!=b.method&&(nn=!0,Pm(Error("AJAX request with postData should use POST")));return e}
function un(a,b,c,d){var e=null;switch(b){case "JSON":try{var f=c.responseText}catch(g){throw d=Error("Error reading responseText"),d.params=a,Qm(d),g;}a=c.getResponseHeader("Content-Type")||"";f&&0<=a.indexOf("json")&&(")]}'\n"===f.substring(0,5)&&(f=f.substring(5)),e=JSON.parse(f));break;case "XML":if(a=(a=c.responseXML)?vn(a):null)e={},fb(a.getElementsByTagName("*"),function(g){e[g.tagName]=wn(g)})}d&&xn(e);
return e}
function xn(a){if(Pa(a))for(var b in a){var c;(c="html_content"==b)||(c=b.length-5,c=0<=c&&b.indexOf("_html",c)==c);if(c){c=b;var d=a[b],e=yb();d=e?e.createHTML(d):d;a[c]=new lc(d)}else xn(a[b])}}
function vn(a){return a?(a=("responseXML"in a?a.responseXML:a).getElementsByTagName("root"))&&0<a.length?a[0]:null:null}
function wn(a){var b="";fb(a.childNodes,function(c){b+=c.nodeValue});
return b}
function yn(a,b){b.method="POST";b.postParams||(b.postParams={});return qn(a,b)}
function tn(a,b,c,d,e,f,g){function h(){4==(l&&"readyState"in l?l.readyState:0)&&b&&Om(b)(l)}
c=void 0===c?"GET":c;d=void 0===d?"":d;var l=hn();if(!l)return null;"onloadend"in l?l.addEventListener("loadend",h,!1):l.onreadystatechange=h;P("debug_forward_web_query_parameters")&&(a=pn(a));l.open(c,a,!0);f&&(l.responseType=f);g&&(l.withCredentials=!0);c="POST"==c&&(void 0===window.FormData||!(d instanceof FormData));if(e=on(a,e))for(var m in e)l.setRequestHeader(m,e[m]),"content-type"==m.toLowerCase()&&(c=!1);c&&l.setRequestHeader("Content-Type","application/x-www-form-urlencoded");l.send(d);
return l}
;var zn=jd||kd;function An(a){var b=Vb();return b?0<=b.toLowerCase().indexOf(a):!1}
;var Bn=[{Ic:function(a){return"Cannot read property '"+a.key+"'"},
lc:{Error:[{regexp:/(Permission denied) to access property "([^']+)"/,groups:["reason","key"]}],TypeError:[{regexp:/Cannot read property '([^']+)' of (null|undefined)/,groups:["key","value"]},{regexp:/\u65e0\u6cd5\u83b7\u53d6\u672a\u5b9a\u4e49\u6216 (null|undefined) \u5f15\u7528\u7684\u5c5e\u6027\u201c([^\u201d]+)\u201d/,groups:["value","key"]},{regexp:/\uc815\uc758\ub418\uc9c0 \uc54a\uc74c \ub610\ub294 (null|undefined) \ucc38\uc870\uc778 '([^']+)' \uc18d\uc131\uc744 \uac00\uc838\uc62c \uc218 \uc5c6\uc2b5\ub2c8\ub2e4./,
groups:["value","key"]},{regexp:/No se puede obtener la propiedad '([^']+)' de referencia nula o sin definir/,groups:["key"]},{regexp:/Unable to get property '([^']+)' of (undefined or null) reference/,groups:["key","value"]},{regexp:/(null) is not an object \(evaluating '(?:([^.]+)\.)?([^']+)'\)/,groups:["value","base","key"]}]}},{Ic:function(a){return"Cannot call '"+a.key+"'"},
lc:{TypeError:[{regexp:/(?:([^ ]+)?\.)?([^ ]+) is not a function/,groups:["base","key"]},{regexp:/([^ ]+) called on (null or undefined)/,groups:["key","value"]},{regexp:/Object (.*) has no method '([^ ]+)'/,groups:["base","key"]},{regexp:/Object doesn't support property or method '([^ ]+)'/,groups:["key"]},{regexp:/\u30aa\u30d6\u30b8\u30a7\u30af\u30c8\u306f '([^']+)' \u30d7\u30ed\u30d1\u30c6\u30a3\u307e\u305f\u306f\u30e1\u30bd\u30c3\u30c9\u3092\u30b5\u30dd\u30fc\u30c8\u3057\u3066\u3044\u307e\u305b\u3093/,
groups:["key"]},{regexp:/\uac1c\uccb4\uac00 '([^']+)' \uc18d\uc131\uc774\ub098 \uba54\uc11c\ub4dc\ub97c \uc9c0\uc6d0\ud558\uc9c0 \uc54a\uc2b5\ub2c8\ub2e4./,groups:["key"]}]}},{Ic:function(a){return a.key+" is not defined"},
lc:{ReferenceError:[{regexp:/(.*) is not defined/,groups:["key"]},{regexp:/Can't find variable: (.*)/,groups:["key"]}]}}];var Dn={Ta:[],Pa:[{callback:Cn,weight:500}]};function Cn(a){if("JavaException"===a.name)return!0;a=a.stack;return a.includes("chrome://")||a.includes("chrome-extension://")||a.includes("moz-extension://")}
;function En(){this.Pa=[];this.Ta=[]}
var Fn;function Gn(){if(!Fn){var a=Fn=new En;a.Ta.length=0;a.Pa.length=0;Dn.Ta&&a.Ta.push.apply(a.Ta,Dn.Ta);Dn.Pa&&a.Pa.push.apply(a.Pa,Dn.Pa)}return Fn}
;var Hn=new $i;function In(a){function b(){return a.charCodeAt(d++)}
var c=a.length,d=0;do{var e=Jn(b);if(Infinity===e)break;var f=e>>3;switch(e&7){case 0:e=Jn(b);if(2===f)return e;break;case 1:if(2===f)return;d+=8;break;case 2:e=Jn(b);if(2===f)return a.substr(d,e);d+=e;break;case 5:if(2===f)return;d+=4;break;default:return}}while(d<c)}
function Jn(a){var b=a(),c=b&127;if(128>b)return c;b=a();c|=(b&127)<<7;if(128>b)return c;b=a();c|=(b&127)<<14;if(128>b)return c;b=a();return 128>b?c|(b&127)<<21:Infinity}
;function Kn(a,b,c,d){if(a)if(Array.isArray(a)){var e=d;for(d=0;d<a.length&&!(a[d]&&(e+=Ln(d,a[d],b,c),500<e));d++);d=e}else if("object"===typeof a)for(e in a){if(a[e]){var f=a[e];var g=b;var h=c;g="string"!==typeof f||"clickTrackingParams"!==e&&"trackingParams"!==e?0:(f=In(atob(f.replace(/-/g,"+").replace(/_/g,"/"))))?Ln(e+".ve",f,g,h):0;d+=g;d+=Ln(e,a[e],b,c);if(500<d)break}}else c[b]=Mn(a),d+=c[b].length;else c[b]=Mn(a),d+=c[b].length;return d}
function Ln(a,b,c,d){c+="."+a;a=Mn(b);d[c]=a;return c.length+a.length}
function Mn(a){try{return("string"===typeof a?a:String(JSON.stringify(a))).substr(0,500)}catch(b){return"unable to serialize "+typeof a+" ("+b.message+")"}}
;function Nn(){this.gf=!0}
function On(){Nn.h||(Nn.h=new Nn);return Nn.h}
function Pn(a,b){a={};var c=Rg([]);c&&(a.Authorization=c,c=b=null==b?void 0:b.sessionIndex,void 0===c&&(c=Number(O("SESSION_INDEX",0)),c=isNaN(c)?0:c),P("voice_search_auth_header_removal")||(a["X-Goog-AuthUser"]=c.toString()),"INNERTUBE_HOST_OVERRIDE"in Jm||(a["X-Origin"]=window.location.origin),void 0===b&&"DELEGATED_SESSION_ID"in Jm&&(a["X-Goog-PageId"]=O("DELEGATED_SESSION_ID")));return a}
;var Qn={identityType:"UNAUTHENTICATED_IDENTITY_TYPE_UNKNOWN"};function Rn(a){var b=this;this.i=void 0;this.h=!1;a.addEventListener("beforeinstallprompt",function(c){c.preventDefault();b.i=c});
a.addEventListener("appinstalled",function(){b.h=!0},{once:!0})}
function Sn(){if(!y.matchMedia)return"WEB_DISPLAY_MODE_UNKNOWN";try{return y.matchMedia("(display-mode: standalone)").matches?"WEB_DISPLAY_MODE_STANDALONE":y.matchMedia("(display-mode: minimal-ui)").matches?"WEB_DISPLAY_MODE_MINIMAL_UI":y.matchMedia("(display-mode: fullscreen)").matches?"WEB_DISPLAY_MODE_FULLSCREEN":y.matchMedia("(display-mode: browser)").matches?"WEB_DISPLAY_MODE_BROWSER":"WEB_DISPLAY_MODE_UNKNOWN"}catch(a){return"WEB_DISPLAY_MODE_UNKNOWN"}}
;function Tn(a,b,c,d,e){Ng.set(""+a,b,{kc:c,path:"/",domain:void 0===d?"youtube.com":d,secure:void 0===e?!1:e})}
function Un(a,b,c){Ng.remove(""+a,void 0===b?"/":b,void 0===c?"youtube.com":c)}
function Vn(){if(!Ng.isEnabled())return!1;if(Ng.h.cookie)return!0;Ng.set("TESTCOOKIESENABLED","1",{kc:60});if("1"!==Ng.get("TESTCOOKIESENABLED"))return!1;Ng.remove("TESTCOOKIESENABLED");return!0}
;var Wn=B("ytglobal.prefsUserPrefsPrefs_")||{};z("ytglobal.prefsUserPrefsPrefs_",Wn);function Xn(){this.h=O("ALT_PREF_COOKIE_NAME","PREF");this.i=O("ALT_PREF_COOKIE_DOMAIN","youtube.com");var a=Ng.get(""+this.h,void 0);a&&this.parse(a)}
var Yn;function Zn(){Yn||(Yn=new Xn);return Yn}
k=Xn.prototype;k.get=function(a,b){$n(a);ao(a);a=void 0!==Wn[a]?Wn[a].toString():null;return null!=a?a:b?b:""};
k.set=function(a,b){$n(a);ao(a);if(null==b)throw Error("ExpectedNotNull");Wn[a]=b.toString()};
function bo(a){return!!((co("f"+(Math.floor(a/31)+1))||0)&1<<a%31)}
k.remove=function(a){$n(a);ao(a);delete Wn[a]};
k.clear=function(){for(var a in Wn)delete Wn[a]};
function ao(a){if(/^f([1-9][0-9]*)$/.test(a))throw Error("ExpectedRegexMatch: "+a);}
function $n(a){if(!/^\w+$/.test(a))throw Error("ExpectedRegexMismatch: "+a);}
function co(a){a=void 0!==Wn[a]?Wn[a].toString():null;return null!=a&&/^[A-Fa-f0-9]+$/.test(a)?parseInt(a,16):null}
k.parse=function(a){a=decodeURIComponent(a).split("&");for(var b=0;b<a.length;b++){var c=a[b].split("="),d=c[0];(c=c[1])&&(Wn[d]=c.toString())}};var eo={bluetooth:"CONN_DISCO",cellular:"CONN_CELLULAR_UNKNOWN",ethernet:"CONN_WIFI",none:"CONN_NONE",wifi:"CONN_WIFI",wimax:"CONN_CELLULAR_4G",other:"CONN_UNKNOWN",unknown:"CONN_UNKNOWN","slow-2g":"CONN_CELLULAR_2G","2g":"CONN_CELLULAR_2G","3g":"CONN_CELLULAR_3G","4g":"CONN_CELLULAR_4G"},fo={CONN_DEFAULT:0,CONN_UNKNOWN:1,CONN_NONE:2,CONN_WIFI:3,CONN_CELLULAR_2G:4,CONN_CELLULAR_3G:5,CONN_CELLULAR_4G:6,CONN_CELLULAR_UNKNOWN:7,CONN_DISCO:8,CONN_CELLULAR_5G:9,CONN_WIFI_METERED:10,CONN_CELLULAR_5G_SA:11,
CONN_CELLULAR_5G_NSA:12,CONN_INVALID:31},go={EFFECTIVE_CONNECTION_TYPE_UNKNOWN:0,EFFECTIVE_CONNECTION_TYPE_OFFLINE:1,EFFECTIVE_CONNECTION_TYPE_SLOW_2G:2,EFFECTIVE_CONNECTION_TYPE_2G:3,EFFECTIVE_CONNECTION_TYPE_3G:4,EFFECTIVE_CONNECTION_TYPE_4G:5},ho={"slow-2g":"EFFECTIVE_CONNECTION_TYPE_SLOW_2G","2g":"EFFECTIVE_CONNECTION_TYPE_2G","3g":"EFFECTIVE_CONNECTION_TYPE_3G","4g":"EFFECTIVE_CONNECTION_TYPE_4G"};function io(){var a=y.navigator;return a?a.connection:void 0}
function jo(){var a=io();if(a){var b=eo[a.type||"unknown"]||"CONN_UNKNOWN";a=eo[a.effectiveType||"unknown"]||"CONN_UNKNOWN";"CONN_CELLULAR_UNKNOWN"===b&&"CONN_UNKNOWN"!==a&&(b=a);if("CONN_UNKNOWN"!==b)return b;if("CONN_UNKNOWN"!==a)return a}}
function ko(){var a=io();if(null!=a&&a.effectiveType)return ho.hasOwnProperty(a.effectiveType)?ho[a.effectiveType]:"EFFECTIVE_CONNECTION_TYPE_UNKNOWN"}
;function lo(a){var b=Ja.apply(1,arguments);var c=Error.call(this,a);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.args=[].concat(ha(b))}
v(lo,Error);function mo(){try{return no(),!0}catch(a){return!1}}
function no(a){if(void 0!==O("DATASYNC_ID"))return O("DATASYNC_ID");throw new lo("Datasync ID not set",void 0===a?"unknown":a);}
;function oo(){}
function po(a,b){return qo(a,0,b)}
oo.prototype.ea=function(a,b){return qo(a,1,b)};
function ro(a){var b=B("yt.scheduler.instance.addImmediateJob");b?b(a):a()}
;function so(){oo.apply(this,arguments)}
v(so,oo);function to(){so.h||(so.h=new so);return so.h}
function qo(a,b,c){void 0!==c&&Number.isNaN(Number(c))&&(c=void 0);var d=B("yt.scheduler.instance.addJob");return d?d(a,b,c):void 0===c?(a(),NaN):kn(a,c||0)}
so.prototype.Ba=function(a){if(void 0===a||!Number.isNaN(Number(a))){var b=B("yt.scheduler.instance.cancelJob");b?b(a):window.clearTimeout(a)}};
so.prototype.start=function(){var a=B("yt.scheduler.instance.start");a&&a()};
so.prototype.pause=function(){var a=B("yt.scheduler.instance.pause");a&&a()};
var Di=to();function uo(a){var b=new kj;(b=b.isAvailable()?a?new qj(b,a):b:null)||(a=new lj(a||"UserDataSharedStore"),b=a.isAvailable()?a:null);this.h=(a=b)?new gj(a):null;this.i=document.domain||window.location.hostname}
uo.prototype.set=function(a,b,c,d){c=c||31104E3;this.remove(a);if(this.h)try{this.h.set(a,b,Date.now()+1E3*c);return}catch(f){}var e="";if(d)try{e=escape(Vg(b))}catch(f){return}else e=escape(b);Tn(a,e,c,this.i)};
uo.prototype.get=function(a,b){var c=void 0,d=!this.h;if(!d)try{c=this.h.get(a)}catch(e){d=!0}if(d&&(c=Ng.get(""+a,void 0))&&(c=unescape(c),b))try{c=JSON.parse(c)}catch(e){this.remove(a),c=void 0}return c};
uo.prototype.remove=function(a){this.h&&this.h.remove(a);Un(a,"/",this.i)};var vo=function(){var a;return function(){a||(a=new uo("ytidb"));return a}}();
function wo(){var a;return null==(a=vo())?void 0:a.get("LAST_RESULT_ENTRY_KEY",!0)}
;var xo=[],yo,zo=!1;function Ao(){var a={};for(yo=new Bo(void 0===a.handleError?Co:a.handleError,void 0===a.logEvent?Do:a.logEvent);0<xo.length;)switch(a=xo.shift(),a.type){case "ERROR":yo.handleError(a.payload);break;case "EVENT":yo.logEvent(a.eventType,a.payload)}}
function Eo(a){zo||(yo?yo.handleError(a):(xo.push({type:"ERROR",payload:a}),10<xo.length&&xo.shift()))}
function Fo(a,b){zo||(yo?yo.logEvent(a,b):(xo.push({type:"EVENT",eventType:a,payload:b}),10<xo.length&&xo.shift()))}
;function Go(a){if(0<=a.indexOf(":"))throw Error("Database name cannot contain ':'");}
function Ho(a){return a.substr(0,a.indexOf(":"))||a}
;var Io={},Jo=(Io.AUTH_INVALID="No user identifier specified.",Io.EXPLICIT_ABORT="Transaction was explicitly aborted.",Io.IDB_NOT_SUPPORTED="IndexedDB is not supported.",Io.MISSING_INDEX="Index not created.",Io.MISSING_OBJECT_STORES="Object stores not created.",Io.DB_DELETED_BY_MISSING_OBJECT_STORES="Database is deleted because expected object stores were not created.",Io.DB_REOPENED_BY_MISSING_OBJECT_STORES="Database is reopened because expected object stores were not created.",Io.UNKNOWN_ABORT="Transaction was aborted for unknown reasons.",
Io.QUOTA_EXCEEDED="The current transaction exceeded its quota limitations.",Io.QUOTA_MAYBE_EXCEEDED="The current transaction may have failed because of exceeding quota limitations.",Io.EXECUTE_TRANSACTION_ON_CLOSED_DB="Can't start a transaction on a closed database",Io.INCOMPATIBLE_DB_VERSION="The binary is incompatible with the database version",Io),Ko={},Lo=(Ko.AUTH_INVALID="ERROR",Ko.EXECUTE_TRANSACTION_ON_CLOSED_DB="WARNING",Ko.EXPLICIT_ABORT="IGNORED",Ko.IDB_NOT_SUPPORTED="ERROR",Ko.MISSING_INDEX=
"WARNING",Ko.MISSING_OBJECT_STORES="ERROR",Ko.DB_DELETED_BY_MISSING_OBJECT_STORES="WARNING",Ko.DB_REOPENED_BY_MISSING_OBJECT_STORES="WARNING",Ko.QUOTA_EXCEEDED="WARNING",Ko.QUOTA_MAYBE_EXCEEDED="WARNING",Ko.UNKNOWN_ABORT="WARNING",Ko.INCOMPATIBLE_DB_VERSION="WARNING",Ko),Mo={},No=(Mo.AUTH_INVALID=!1,Mo.EXECUTE_TRANSACTION_ON_CLOSED_DB=!1,Mo.EXPLICIT_ABORT=!1,Mo.IDB_NOT_SUPPORTED=!1,Mo.MISSING_INDEX=!1,Mo.MISSING_OBJECT_STORES=!1,Mo.DB_DELETED_BY_MISSING_OBJECT_STORES=!1,Mo.DB_REOPENED_BY_MISSING_OBJECT_STORES=
!1,Mo.QUOTA_EXCEEDED=!1,Mo.QUOTA_MAYBE_EXCEEDED=!0,Mo.UNKNOWN_ABORT=!0,Mo.INCOMPATIBLE_DB_VERSION=!1,Mo);function Oo(a,b,c,d,e){b=void 0===b?{}:b;c=void 0===c?Jo[a]:c;d=void 0===d?Lo[a]:d;e=void 0===e?No[a]:e;lo.call(this,c,Object.assign({},{name:"YtIdbKnownError",isSw:void 0===self.document,isIframe:self!==self.top,type:a},b));this.type=a;this.message=c;this.level=d;this.h=e;Object.setPrototypeOf(this,Oo.prototype)}
v(Oo,lo);function Po(a,b){Oo.call(this,"MISSING_OBJECT_STORES",{expectedObjectStores:b,foundObjectStores:a},Jo.MISSING_OBJECT_STORES);Object.setPrototypeOf(this,Po.prototype)}
v(Po,Oo);function Qo(a,b){var c=Error.call(this);this.message=c.message;"stack"in c&&(this.stack=c.stack);this.index=a;this.objectStore=b;Object.setPrototypeOf(this,Qo.prototype)}
v(Qo,Error);var Ro=["The database connection is closing","Can't start a transaction on a closed database","A mutation operation was attempted on a database that did not allow mutations"];
function So(a,b,c,d){b=Ho(b);var e=a instanceof Error?a:Error("Unexpected error: "+a);if(e instanceof Oo)return e;a={objectStoreNames:c,dbName:b,dbVersion:d};if("QuotaExceededError"===e.name)return new Oo("QUOTA_EXCEEDED",a);if(ld&&"UnknownError"===e.name)return new Oo("QUOTA_MAYBE_EXCEEDED",a);if(e instanceof Qo)return new Oo("MISSING_INDEX",Object.assign({},a,{objectStore:e.objectStore,index:e.index}));if("InvalidStateError"===e.name&&Ro.some(function(f){return e.message.includes(f)}))return new Oo("EXECUTE_TRANSACTION_ON_CLOSED_DB",
a);
if("AbortError"===e.name)return new Oo("UNKNOWN_ABORT",a,e.message);e.args=[Object.assign({},a,{name:"IdbError",Cd:e.name})];e.level="WARNING";return e}
function To(a,b,c){var d=wo();return new Oo("IDB_NOT_SUPPORTED",{context:{caller:a,publicName:b,version:c,hasSucceededOnce:null==d?void 0:d.hasSucceededOnce}})}
;function Uo(a){if(!a)throw Error();throw a;}
function Vo(a){return a}
function Wo(a){this.h=a}
function Xo(a){function b(e){if("PENDING"===d.state.status){d.state={status:"REJECTED",reason:e};e=r(d.i);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
function c(e){if("PENDING"===d.state.status){d.state={status:"FULFILLED",value:e};e=r(d.h);for(var f=e.next();!f.done;f=e.next())f=f.value,f()}}
var d=this;this.state={status:"PENDING"};this.h=[];this.i=[];a=a.h;try{a(c,b)}catch(e){b(e)}}
Xo.all=function(a){return new Xo(new Wo(function(b,c){var d=[],e=a.length;0===e&&b(d);for(var f={nb:0};f.nb<a.length;f={nb:f.nb},++f.nb)Xo.resolve(a[f.nb]).then(function(g){return function(h){d[g.nb]=h;e--;0===e&&b(d)}}(f)).catch(function(g){c(g)})}))};
Xo.resolve=function(a){return new Xo(new Wo(function(b,c){a instanceof Xo?a.then(b,c):b(a)}))};
Xo.reject=function(a){return new Xo(new Wo(function(b,c){c(a)}))};
Xo.prototype.then=function(a,b){var c=this,d=null!=a?a:Vo,e=null!=b?b:Uo;return new Xo(new Wo(function(f,g){"PENDING"===c.state.status?(c.h.push(function(){Yo(c,c,d,f,g)}),c.i.push(function(){Zo(c,c,e,f,g)})):"FULFILLED"===c.state.status?Yo(c,c,d,f,g):"REJECTED"===c.state.status&&Zo(c,c,e,f,g)}))};
Xo.prototype.catch=function(a){return this.then(void 0,a)};
function Yo(a,b,c,d,e){try{if("FULFILLED"!==a.state.status)throw Error("calling handleResolve before the promise is fulfilled.");var f=c(a.state.value);f instanceof Xo?$o(a,b,f,d,e):d(f)}catch(g){e(g)}}
function Zo(a,b,c,d,e){try{if("REJECTED"!==a.state.status)throw Error("calling handleReject before the promise is rejected.");var f=c(a.state.reason);f instanceof Xo?$o(a,b,f,d,e):d(f)}catch(g){e(g)}}
function $o(a,b,c,d,e){b===c?e(new TypeError("Circular promise chain detected.")):c.then(function(f){f instanceof Xo?$o(a,b,f,d,e):d(f)},function(f){e(f)})}
;function ap(a,b,c){function d(){c(a.error);f()}
function e(){b(a.result);f()}
function f(){try{a.removeEventListener("success",e),a.removeEventListener("error",d)}catch(g){}}
a.addEventListener("success",e);a.addEventListener("error",d)}
function bp(a){return new Promise(function(b,c){ap(a,b,c)})}
function cp(a){return new Xo(new Wo(function(b,c){ap(a,b,c)}))}
;function dp(a,b){return new Xo(new Wo(function(c,d){function e(){var f=a?b(a):null;f?f.then(function(g){a=g;e()},d):c()}
e()}))}
;var ep=window,R=ep.ytcsi&&ep.ytcsi.now?ep.ytcsi.now:ep.performance&&ep.performance.timing&&ep.performance.now&&ep.performance.timing.navigationStart?function(){return ep.performance.timing.navigationStart+ep.performance.now()}:function(){return(new Date).getTime()};function fp(a,b){this.h=a;this.options=b;this.transactionCount=0;this.j=Math.round(R());this.i=!1}
k=fp.prototype;k.add=function(a,b,c){return gp(this,[a],{mode:"readwrite",ca:!0},function(d){return d.objectStore(a).add(b,c)})};
k.clear=function(a){return gp(this,[a],{mode:"readwrite",ca:!0},function(b){return b.objectStore(a).clear()})};
k.close=function(){this.h.close();var a;(null==(a=this.options)?0:a.closed)&&this.options.closed()};
k.count=function(a,b){return gp(this,[a],{mode:"readonly",ca:!0},function(c){return c.objectStore(a).count(b)})};
function hp(a,b,c){a=a.h.createObjectStore(b,c);return new ip(a)}
k.delete=function(a,b){return gp(this,[a],{mode:"readwrite",ca:!0},function(c){return c.objectStore(a).delete(b)})};
k.get=function(a,b){return gp(this,[a],{mode:"readonly",ca:!0},function(c){return c.objectStore(a).get(b)})};
function jp(a,b,c){return gp(a,[b],{mode:"readwrite",ca:!0},function(d){d=d.objectStore(b);return cp(d.h.put(c,void 0))})}
k.objectStoreNames=function(){return Array.from(this.h.objectStoreNames)};
function gp(a,b,c,d){var e,f,g,h,l,m,n,q,u,t,A,C;return x(function(E){switch(E.h){case 1:var M={mode:"readonly",ca:!1,tag:"IDB_TRANSACTION_TAG_UNKNOWN"};"string"===typeof c?M.mode=c:Object.assign(M,c);e=M;a.transactionCount++;f=e.ca?3:1;g=0;case 2:if(h){E.A(3);break}g++;l=Math.round(R());xa(E,4);m=a.h.transaction(b,e.mode);M=new kp(m);M=lp(M,d);return w(E,M,6);case 6:return n=E.i,q=Math.round(R()),mp(a,l,q,g,void 0,b.join(),e),E.return(n);case 4:u=za(E);t=Math.round(R());A=So(u,a.h.name,b.join(),
a.h.version);if((C=A instanceof Oo&&!A.h)||g>=f)mp(a,l,t,g,A,b.join(),e),h=A;E.A(2);break;case 3:return E.return(Promise.reject(h))}})}
function mp(a,b,c,d,e,f,g){b=c-b;e?(e instanceof Oo&&("QUOTA_EXCEEDED"===e.type||"QUOTA_MAYBE_EXCEEDED"===e.type)&&Fo("QUOTA_EXCEEDED",{dbName:Ho(a.h.name),objectStoreNames:f,transactionCount:a.transactionCount,transactionMode:g.mode}),e instanceof Oo&&"UNKNOWN_ABORT"===e.type&&(c-=a.j,0>c&&c>=Math.pow(2,31)&&(c=0),Fo("TRANSACTION_UNEXPECTEDLY_ABORTED",{objectStoreNames:f,transactionDuration:b,transactionCount:a.transactionCount,dbDuration:c}),a.i=!0),np(a,!1,d,f,b,g.tag),Eo(e)):np(a,!0,d,f,b,g.tag)}
function np(a,b,c,d,e,f){Fo("TRANSACTION_ENDED",{objectStoreNames:d,connectionHasUnknownAbortedTransaction:a.i,duration:e,isSuccessful:b,tryCount:c,tag:void 0===f?"IDB_TRANSACTION_TAG_UNKNOWN":f})}
k.getName=function(){return this.h.name};
function ip(a){this.h=a}
k=ip.prototype;k.add=function(a,b){return cp(this.h.add(a,b))};
k.autoIncrement=function(){return this.h.autoIncrement};
k.clear=function(){return cp(this.h.clear()).then(function(){})};
function op(a,b,c){a.h.createIndex(b,c,{unique:!1})}
k.count=function(a){return cp(this.h.count(a))};
function pp(a,b){return qp(a,{query:b},function(c){return c.delete().then(function(){return c.continue()})}).then(function(){})}
k.delete=function(a){return a instanceof IDBKeyRange?pp(this,a):cp(this.h.delete(a))};
k.get=function(a){return cp(this.h.get(a))};
k.index=function(a){try{return new rp(this.h.index(a))}catch(b){if(b instanceof Error&&"NotFoundError"===b.name)throw new Qo(a,this.h.name);throw b;}};
k.getName=function(){return this.h.name};
k.keyPath=function(){return this.h.keyPath};
function qp(a,b,c){a=a.h.openCursor(b.query,b.direction);return sp(a).then(function(d){return dp(d,c)})}
function kp(a){var b=this;this.h=a;this.j=new Map;this.i=!1;this.done=new Promise(function(c,d){b.h.addEventListener("complete",function(){c()});
b.h.addEventListener("error",function(e){e.currentTarget===e.target&&d(b.h.error)});
b.h.addEventListener("abort",function(){var e=b.h.error;if(e)d(e);else if(!b.i){e=Oo;for(var f=b.h.objectStoreNames,g=[],h=0;h<f.length;h++){var l=f.item(h);if(null===l)throw Error("Invariant: item in DOMStringList is null");g.push(l)}e=new e("UNKNOWN_ABORT",{objectStoreNames:g.join(),dbName:b.h.db.name,mode:b.h.mode});d(e)}})})}
function lp(a,b){var c=new Promise(function(d,e){try{b(a).then(function(f){d(f)}).catch(e)}catch(f){e(f),a.abort()}});
return Promise.all([c,a.done]).then(function(d){return r(d).next().value})}
kp.prototype.abort=function(){this.h.abort();this.i=!0;throw new Oo("EXPLICIT_ABORT");};
kp.prototype.objectStore=function(a){a=this.h.objectStore(a);var b=this.j.get(a);b||(b=new ip(a),this.j.set(a,b));return b};
function rp(a){this.h=a}
k=rp.prototype;k.count=function(a){return cp(this.h.count(a))};
k.delete=function(a){return tp(this,{query:a},function(b){return b.delete().then(function(){return b.continue()})})};
k.get=function(a){return cp(this.h.get(a))};
k.getKey=function(a){return cp(this.h.getKey(a))};
k.keyPath=function(){return this.h.keyPath};
k.unique=function(){return this.h.unique};
function tp(a,b,c){a=a.h.openCursor(void 0===b.query?null:b.query,void 0===b.direction?"next":b.direction);return sp(a).then(function(d){return dp(d,c)})}
function up(a,b){this.request=a;this.cursor=b}
function sp(a){return cp(a).then(function(b){return b?new up(a,b):null})}
k=up.prototype;k.advance=function(a){this.cursor.advance(a);return sp(this.request)};
k.continue=function(a){this.cursor.continue(a);return sp(this.request)};
k.delete=function(){return cp(this.cursor.delete()).then(function(){})};
k.getKey=function(){return this.cursor.key};
k.getValue=function(){return this.cursor.value};
k.update=function(a){return cp(this.cursor.update(a))};function vp(a,b,c){return new Promise(function(d,e){function f(){u||(u=new fp(g.result,{closed:q}));return u}
var g=void 0!==b?self.indexedDB.open(a,b):self.indexedDB.open(a);var h=c.de,l=c.blocking,m=c.hf,n=c.upgrade,q=c.closed,u;g.addEventListener("upgradeneeded",function(t){try{if(null===t.newVersion)throw Error("Invariant: newVersion on IDbVersionChangeEvent is null");if(null===g.transaction)throw Error("Invariant: transaction on IDbOpenDbRequest is null");t.dataLoss&&"none"!==t.dataLoss&&Fo("IDB_DATA_CORRUPTED",{reason:t.dataLossMessage||"unknown reason",dbName:Ho(a)});var A=f(),C=new kp(g.transaction);
n&&n(A,function(E){return t.oldVersion<E&&t.newVersion>=E},C);
C.done.catch(function(E){e(E)})}catch(E){e(E)}});
g.addEventListener("success",function(){var t=g.result;l&&t.addEventListener("versionchange",function(){l(f())});
t.addEventListener("close",function(){Fo("IDB_UNEXPECTEDLY_CLOSED",{dbName:Ho(a),dbVersion:t.version});m&&m()});
d(f())});
g.addEventListener("error",function(){e(g.error)});
h&&g.addEventListener("blocked",function(){h()})})}
function wp(a,b,c){c=void 0===c?{}:c;return vp(a,b,c)}
function xp(a,b){b=void 0===b?{}:b;var c,d,e,f;return x(function(g){if(1==g.h)return xa(g,2),c=self.indexedDB.deleteDatabase(a),d=b,(e=d.de)&&c.addEventListener("blocked",function(){e()}),w(g,bp(c),4);
if(2!=g.h)return ya(g,0);f=za(g);throw So(f,a,"",-1);})}
;function yp(a,b){this.name=a;this.options=b;this.j=!0;this.m=this.l=0}
yp.prototype.i=function(a,b,c){c=void 0===c?{}:c;return wp(a,b,c)};
yp.prototype.delete=function(a){a=void 0===a?{}:a;return xp(this.name,a)};
function zp(a,b){return new Oo("INCOMPATIBLE_DB_VERSION",{dbName:a.name,oldVersion:a.options.version,newVersion:b})}
function Ap(a,b){if(!b)throw To("openWithToken",Ho(a.name));return Bp(a)}
function Bp(a){function b(){var f,g,h,l,m,n,q,u,t,A;return x(function(C){switch(C.h){case 1:return g=null!=(f=Error().stack)?f:"",xa(C,2),w(C,a.i(a.name,a.options.version,d),4);case 4:h=C.i;for(var E=a.options,M=[],Q=r(Object.keys(E.Ab)),S=Q.next();!S.done;S=Q.next()){S=S.value;var V=E.Ab[S],Z=void 0===V.Pe?Number.MAX_VALUE:V.Pe;!(h.h.version>=V.Gb)||h.h.version>=Z||h.h.objectStoreNames.contains(S)||M.push(S)}l=M;if(0===l.length){C.A(5);break}m=Object.keys(a.options.Ab);n=h.objectStoreNames();if(a.m<
dn("ytidb_reopen_db_retries",0))return a.m++,h.close(),Eo(new Oo("DB_REOPENED_BY_MISSING_OBJECT_STORES",{dbName:a.name,expectedObjectStores:m,foundObjectStores:n})),C.return(b());if(!(a.l<dn("ytidb_remake_db_retries",1))){C.A(6);break}a.l++;return w(C,a.delete(),7);case 7:return Eo(new Oo("DB_DELETED_BY_MISSING_OBJECT_STORES",{dbName:a.name,expectedObjectStores:m,foundObjectStores:n})),C.return(b());case 6:throw new Po(n,m);case 5:return C.return(h);case 2:q=za(C);if(q instanceof DOMException?"VersionError"!==
q.name:"DOMError"in self&&q instanceof DOMError?"VersionError"!==q.name:!(q instanceof Object&&"message"in q)||"An attempt was made to open a database using a lower version than the existing version."!==q.message){C.A(8);break}return w(C,a.i(a.name,void 0,Object.assign({},d,{upgrade:void 0})),9);case 9:u=C.i;t=u.h.version;if(void 0!==a.options.version&&t>a.options.version+1)throw u.close(),a.j=!1,zp(a,t);return C.return(u);case 8:throw c(),q instanceof Error&&!P("ytidb_async_stack_killswitch")&&(q.stack=
q.stack+"\n"+g.substring(g.indexOf("\n")+1)),So(q,a.name,"",null!=(A=a.options.version)?A:-1);}})}
function c(){a.h===e&&(a.h=void 0)}
if(!a.j)throw zp(a);if(a.h)return a.h;var d={blocking:function(f){f.close()},
closed:c,hf:c,upgrade:a.options.upgrade};var e=b();a.h=e;return a.h}
;var Cp=new yp("YtIdbMeta",{Ab:{databases:{Gb:1}},upgrade:function(a,b){b(1)&&hp(a,"databases",{keyPath:"actualName"})}});
function Dp(a,b){var c;return x(function(d){if(1==d.h)return w(d,Ap(Cp,b),2);c=d.i;return d.return(gp(c,["databases"],{ca:!0,mode:"readwrite"},function(e){var f=e.objectStore("databases");return f.get(a.actualName).then(function(g){if(g?a.actualName!==g.actualName||a.publicName!==g.publicName||a.userIdentifier!==g.userIdentifier:1)return cp(f.h.put(a,void 0)).then(function(){})})}))})}
function Ep(a,b){var c;return x(function(d){if(1==d.h)return a?w(d,Ap(Cp,b),2):d.return();c=d.i;return d.return(c.delete("databases",a))})}
function Fp(a,b){var c,d;return x(function(e){return 1==e.h?(c=[],w(e,Ap(Cp,b),2)):3!=e.h?(d=e.i,w(e,gp(d,["databases"],{ca:!0,mode:"readonly"},function(f){c.length=0;return qp(f.objectStore("databases"),{},function(g){a(g.getValue())&&c.push(g.getValue());return g.continue()})}),3)):e.return(c)})}
function Gp(a){return Fp(function(b){return"LogsDatabaseV2"===b.publicName&&void 0!==b.userIdentifier},a)}
function Hp(a,b,c){return Fp(function(d){return c?void 0!==d.userIdentifier&&!a.includes(d.userIdentifier)&&c.includes(d.publicName):void 0!==d.userIdentifier&&!a.includes(d.userIdentifier)},b)}
function Ip(a){var b,c;return x(function(d){if(1==d.h)return b=no("YtIdbMeta hasAnyMeta other"),w(d,Fp(function(e){return void 0!==e.userIdentifier&&e.userIdentifier!==b},a),2);
c=d.i;return d.return(0<c.length)})}
;var Jp,Kp=new function(){}(new function(){});
function Lp(){var a,b,c,d;return x(function(e){switch(e.h){case 1:a=wo();if(null==(b=a)?0:b.hasSucceededOnce)return e.return(!0);var f;if(f=zn)f=/WebKit\/([0-9]+)/.exec(Vb()),f=!!(f&&600<=parseInt(f[1],10));f&&(f=/WebKit\/([0-9]+)/.exec(Vb()),f=!(f&&602<=parseInt(f[1],10)));if(f||Xc)return e.return(!1);try{if(c=self,!(c.indexedDB&&c.IDBIndex&&c.IDBKeyRange&&c.IDBObjectStore))return e.return(!1)}catch(g){return e.return(!1)}if(!("IDBTransaction"in self&&"objectStoreNames"in IDBTransaction.prototype))return e.return(!1);
xa(e,2);d={actualName:"yt-idb-test-do-not-use",publicName:"yt-idb-test-do-not-use",userIdentifier:void 0};return w(e,Dp(d,Kp),4);case 4:return w(e,Ep("yt-idb-test-do-not-use",Kp),5);case 5:return e.return(!0);case 2:return za(e),e.return(!1)}})}
function Mp(){if(void 0!==Jp)return Jp;zo=!0;return Jp=Lp().then(function(a){zo=!1;var b;if(null!=(b=vo())&&b.h){var c;b={hasSucceededOnce:(null==(c=wo())?void 0:c.hasSucceededOnce)||a};var d;null==(d=vo())||d.set("LAST_RESULT_ENTRY_KEY",b,2592E3,!0)}return a})}
function Np(){return B("ytglobal.idbToken_")||void 0}
function Op(){var a=Np();return a?Promise.resolve(a):Mp().then(function(b){(b=b?Kp:void 0)&&z("ytglobal.idbToken_",b);return b})}
;var Pp=0;function Qp(a,b){Pp||(Pp=Di.ea(function(){var c,d,e,f,g;return x(function(h){switch(h.h){case 1:return w(h,Op(),2);case 2:c=h.i;if(!c)return h.return();d=!0;xa(h,3);return w(h,Hp(a,c,b),5);case 5:e=h.i;if(!e.length){d=!1;h.A(6);break}f=e[0];return w(h,xp(f.actualName),7);case 7:return w(h,Ep(f.actualName,c),6);case 6:ya(h,4);break;case 3:g=za(h),Eo(g),d=!1;case 4:Di.Ba(Pp),Pp=0,d&&Qp(a,b),h.h=0}})}))}
function Rp(){var a;return x(function(b){return 1==b.h?w(b,Op(),2):(a=b.i)?b.return(Ip(a)):b.return(!1)})}
new $h;function Sp(a){if(!mo())throw a=new Oo("AUTH_INVALID",{dbName:a}),Eo(a),a;var b=no();return{actualName:a+":"+b,publicName:a,userIdentifier:b}}
function Tp(a,b,c,d){var e,f,g,h,l,m;return x(function(n){switch(n.h){case 1:return f=null!=(e=Error().stack)?e:"",w(n,Op(),2);case 2:g=n.i;if(!g)throw h=To("openDbImpl",a,b),P("ytidb_async_stack_killswitch")||(h.stack=h.stack+"\n"+f.substring(f.indexOf("\n")+1)),Eo(h),h;Go(a);l=c?{actualName:a,publicName:a,userIdentifier:void 0}:Sp(a);xa(n,3);return w(n,Dp(l,g),5);case 5:return w(n,wp(l.actualName,b,d),6);case 6:return n.return(n.i);case 3:return m=za(n),xa(n,7),w(n,Ep(l.actualName,g),9);case 9:ya(n,
8);break;case 7:za(n);case 8:throw m;}})}
function Up(a,b,c){c=void 0===c?{}:c;return Tp(a,b,!1,c)}
function Vp(a,b,c){c=void 0===c?{}:c;return Tp(a,b,!0,c)}
function Wp(a,b){b=void 0===b?{}:b;var c,d;return x(function(e){if(1==e.h)return w(e,Op(),2);if(3!=e.h){c=e.i;if(!c)return e.return();Go(a);d=Sp(a);return w(e,xp(d.actualName,b),3)}return w(e,Ep(d.actualName,c),0)})}
function Xp(a,b,c){a=a.map(function(d){return x(function(e){return 1==e.h?w(e,xp(d.actualName,b),2):w(e,Ep(d.actualName,c),0)})});
return Promise.all(a).then(function(){})}
function Yp(){var a=void 0===a?{}:a;var b,c;return x(function(d){if(1==d.h)return w(d,Op(),2);if(3!=d.h){b=d.i;if(!b)return d.return();Go("LogsDatabaseV2");return w(d,Gp(b),3)}c=d.i;return w(d,Xp(c,a,b),0)})}
function Zp(a,b){b=void 0===b?{}:b;var c;return x(function(d){if(1==d.h)return w(d,Op(),2);if(3!=d.h){c=d.i;if(!c)return d.return();Go(a);return w(d,xp(a,b),3)}return w(d,Ep(a,c),0)})}
;function $p(a,b){yp.call(this,a,b);this.options=b;Go(a)}
v($p,yp);function aq(a,b){var c;return function(){c||(c=new $p(a,b));return c}}
$p.prototype.i=function(a,b,c){c=void 0===c?{}:c;return(this.options.qc?Vp:Up)(a,b,Object.assign({},c))};
$p.prototype.delete=function(a){a=void 0===a?{}:a;return(this.options.qc?Zp:Wp)(this.name,a)};
function bq(a,b){return aq(a,b)}
;var cq={},dq=bq("ytGcfConfig",{Ab:(cq.coldConfigStore={Gb:1},cq.hotConfigStore={Gb:1},cq),qc:!1,upgrade:function(a,b){b(1)&&(op(hp(a,"hotConfigStore",{keyPath:"key",autoIncrement:!0}),"hotTimestampIndex","timestamp"),op(hp(a,"coldConfigStore",{keyPath:"key",autoIncrement:!0}),"coldTimestampIndex","timestamp"))},
version:1});function eq(a){return Ap(dq(),a)}
function fq(a,b,c){var d,e,f;return x(function(g){switch(g.h){case 1:return d={config:a,hashData:b,timestamp:R()},w(g,eq(c),2);case 2:return e=g.i,w(g,e.clear("hotConfigStore"),3);case 3:return w(g,jp(e,"hotConfigStore",d),4);case 4:return f=g.i,g.return(f)}})}
function gq(a,b,c,d){var e,f,g;return x(function(h){switch(h.h){case 1:return e={config:a,hashData:b,configData:c,timestamp:R()},w(h,eq(d),2);case 2:return f=h.i,w(h,f.clear("coldConfigStore"),3);case 3:return w(h,jp(f,"coldConfigStore",e),4);case 4:return g=h.i,h.return(g)}})}
function hq(a){var b,c;return x(function(d){return 1==d.h?w(d,eq(a),2):3!=d.h?(b=d.i,c=void 0,w(d,gp(b,["coldConfigStore"],{mode:"readwrite",ca:!0},function(e){return tp(e.objectStore("coldConfigStore").index("coldTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
function iq(a){var b,c;return x(function(d){return 1==d.h?w(d,eq(a),2):3!=d.h?(b=d.i,c=void 0,w(d,gp(b,["hotConfigStore"],{mode:"readwrite",ca:!0},function(e){return tp(e.objectStore("hotConfigStore").index("hotTimestampIndex"),{direction:"prev"},function(f){c=f.getValue()})}),3)):d.return(c)})}
;function jq(){this.h=0}
function kq(a,b,c){var d,e,f;return x(function(g){if(1==g.h){if(!P("update_log_event_config"))return g.A(0);c&&(a.i=c,z("yt.gcf.config.hotConfigGroup",a.i));a.hotHashData=b;z("yt.gcf.config.hotHashData",a.hotHashData);return(d=Np())?c?g.A(4):w(g,iq(d),5):g.A(0)}4!=g.h&&(e=g.i,c=null==(f=e)?void 0:f.config);return w(g,fq(c,b,d),0)})}
function lq(a,b,c){var d,e,f,g;return x(function(h){if(1==h.h){if(!P("update_log_event_config"))return h.A(0);a.coldHashData=b;z("yt.gcf.config.coldHashData",a.coldHashData);return(d=Np())?c?h.A(4):w(h,hq(d),5):h.A(0)}4!=h.h&&(e=h.i,c=null==(f=e)?void 0:f.config);if(!c)return h.A(0);g=c.configData;return w(h,gq(c,b,g,d),0)})}
;function mq(){return"INNERTUBE_API_KEY"in Jm&&"INNERTUBE_API_VERSION"in Jm}
function nq(){return{innertubeApiKey:O("INNERTUBE_API_KEY"),innertubeApiVersion:O("INNERTUBE_API_VERSION"),Dc:O("INNERTUBE_CONTEXT_CLIENT_CONFIG_INFO"),td:O("INNERTUBE_CONTEXT_CLIENT_NAME","WEB"),ye:O("INNERTUBE_CONTEXT_CLIENT_NAME",1),innertubeContextClientVersion:O("INNERTUBE_CONTEXT_CLIENT_VERSION"),vd:O("INNERTUBE_CONTEXT_HL"),ud:O("INNERTUBE_CONTEXT_GL"),ze:O("INNERTUBE_HOST_OVERRIDE")||"",Be:!!O("INNERTUBE_USE_THIRD_PARTY_AUTH",!1),Ae:!!O("INNERTUBE_OMIT_API_KEY_WHEN_AUTH_HEADER_IS_PRESENT",
!1),appInstallData:O("SERIALIZED_CLIENT_CONFIG_DATA")}}
function oq(a){var b={client:{hl:a.vd,gl:a.ud,clientName:a.td,clientVersion:a.innertubeContextClientVersion,configInfo:a.Dc}};navigator.userAgent&&(b.client.userAgent=String(navigator.userAgent));var c=y.devicePixelRatio;c&&1!=c&&(b.client.screenDensityFloat=String(c));c=en();""!==c&&(b.client.experimentsToken=c);c=fn();0<c.length&&(b.request={internalExperimentFlags:c});pq(a,void 0,b);qq(void 0,b);rq(void 0,b);sq(a,void 0,b);tq(void 0,b);P("start_sending_config_hash")&&uq(void 0,b);O("DELEGATED_SESSION_ID")&&
!P("pageid_as_header_web")&&(b.user={onBehalfOfUser:O("DELEGATED_SESSION_ID")});a=Object;c=a.assign;for(var d=b.client,e={},f=r(Object.entries(Xm(O("DEVICE","")))),g=f.next();!g.done;g=f.next()){var h=r(g.value);g=h.next().value;h=h.next().value;"cbrand"===g?e.deviceMake=h:"cmodel"===g?e.deviceModel=h:"cbr"===g?e.browserName=h:"cbrver"===g?e.browserVersion=h:"cos"===g?e.osName=h:"cosver"===g?e.osVersion=h:"cplatform"===g&&(e.platform=h)}b.client=c.call(a,d,e);return b}
function vq(a){var b=new dl,c=new Tk;G(c,1,a.vd);G(c,2,a.ud);G(c,16,a.ye);G(c,17,a.innertubeContextClientVersion);if(a.Dc){var d=a.Dc,e=new Pk;d.coldConfigData&&G(e,1,d.coldConfigData);d.appInstallData&&G(e,6,d.appInstallData);d.coldHashData&&G(e,3,d.coldHashData);d.hotHashData&&G(e,5,d.hotHashData);H(c,Pk,62,e)}(d=y.devicePixelRatio)&&1!=d&&G(c,65,Ud(d));d=en();""!==d&&G(c,54,d);d=fn();if(0<d.length){e=new Xk;for(var f=0;f<d.length;f++){var g=new Vk;G(g,1,d[f].key);ee(g,2,Wk,d[f].value);le(e,15,
Vk,g)}H(b,Xk,5,e)}pq(a,c);qq(b);rq(c);sq(a,c);tq(c);P("start_sending_config_hash")&&uq(c);O("DELEGATED_SESSION_ID")&&!P("pageid_as_header_web")&&(a=new bl,G(a,3,O("DELEGATED_SESSION_ID")));a=r(Object.entries(Xm(O("DEVICE",""))));for(d=a.next();!d.done;d=a.next())e=r(d.value),d=e.next().value,e=e.next().value,"cbrand"===d?G(c,12,e):"cmodel"===d?G(c,13,e):"cbr"===d?G(c,87,e):"cbrver"===d?G(c,88,e):"cos"===d?G(c,18,e):"cosver"===d?G(c,19,e):"cplatform"===d&&G(c,42,e);b.l(c);return b}
function pq(a,b,c){a=a.td;if("WEB"===a||"MWEB"===a||1===a||2===a)if(b){c=ge(b,Qk,96)||new Qk;var d=Sn();d=Object.keys(Bl).indexOf(d);d=-1===d?null:d;null!==d&&G(c,3,d);H(b,Qk,96,c)}else c&&(c.client.mainAppWebInfo=null!=(d=c.client.mainAppWebInfo)?d:{},c.client.mainAppWebInfo.webDisplayMode=Sn())}
function qq(a,b){var c=B("yt.embedded_player.embed_url");c&&(a?(b=ge(a,Zk,7)||new Zk,G(b,4,c),H(a,Zk,7,b)):b&&(b.thirdParty={embedUrl:c}))}
function rq(a,b){var c;if(P("web_log_memory_total_kbytes")&&(null==(c=y.navigator)?0:c.deviceMemory)){var d;c=null==(d=y.navigator)?void 0:d.deviceMemory;a?G(a,95,1E6*c):b&&(b.client.memoryTotalKbytes=""+1E6*c)}}
function sq(a,b,c){if(a.appInstallData)if(b){var d;c=null!=(d=ge(b,Pk,62))?d:new Pk;G(c,6,a.appInstallData);H(b,Pk,62,c)}else c&&(c.client.configInfo=c.client.configInfo||{},c.client.configInfo.appInstallData=a.appInstallData)}
function tq(a,b){var c=jo();c&&(a?G(a,61,fo[c]):b&&(b.client.connectionType=c));P("web_log_effective_connection_type")&&(c=ko())&&(a?G(a,94,go[c]):b&&(b.client.effectiveConnectionType=c))}
function wq(a,b,c){c=void 0===c?{}:c;var d={};O("EOM_VISITOR_DATA")?d={"X-Goog-EOM-Visitor-Id":O("EOM_VISITOR_DATA")}:d={"X-Goog-Visitor-Id":c.visitorData||O("VISITOR_DATA","")};if(b&&b.includes("www.youtube-nocookie.com"))return d;b=c.og||O("AUTHORIZATION");b||(a?b="Bearer "+B("gapi.auth.getToken")().ng:(a=Pn(On()),P("pageid_as_header_web")||delete a["X-Goog-PageId"],d=Object.assign({},d,a)));b&&(d.Authorization=b);return d}
function uq(a,b){jq.h||(jq.h=new jq);var c=jq.h;var d=R()-c.h;if(0!==c.h&&d<dn("send_config_hash_timer"))c=void 0;else{d=B("yt.gcf.config.coldConfigData");var e=B("yt.gcf.config.hotHashData"),f=B("yt.gcf.config.coldHashData");d&&e&&f&&(c.h=R());c={coldConfigData:d,hotHashData:e,coldHashData:f}}if(e=c)if(c=e.coldConfigData,d=e.coldHashData,e=e.hotHashData,c&&d&&e)if(a){var g;b=null!=(g=ge(a,Pk,62))?g:new Pk;G(b,1,c);G(b,3,d);G(b,5,e);H(a,Pk,62,b)}else b&&(b.client.configInfo=b.client.configInfo||{},
b.client.configInfo.coldConfigData=c,b.client.configInfo.coldHashData=d,b.client.configInfo.hotHashData=e)}
;function xq(a,b){this.version=a;this.args=b}
;function yq(a,b){this.topic=a;this.h=b}
yq.prototype.toString=function(){return this.topic};var zq=B("ytPubsub2Pubsub2Instance")||new $i;$i.prototype.subscribe=$i.prototype.subscribe;$i.prototype.unsubscribeByKey=$i.prototype.Fb;$i.prototype.publish=$i.prototype.cb;$i.prototype.clear=$i.prototype.clear;z("ytPubsub2Pubsub2Instance",zq);var Aq=B("ytPubsub2Pubsub2SubscribedKeys")||{};z("ytPubsub2Pubsub2SubscribedKeys",Aq);var Bq=B("ytPubsub2Pubsub2TopicToKeys")||{};z("ytPubsub2Pubsub2TopicToKeys",Bq);var Cq=B("ytPubsub2Pubsub2IsAsync")||{};z("ytPubsub2Pubsub2IsAsync",Cq);
z("ytPubsub2Pubsub2SkipSubKey",null);function Dq(a,b){var c=Eq();c&&c.publish.call(c,a.toString(),a,b)}
function Fq(a){var b=Gq,c=Eq();if(!c)return 0;var d=c.subscribe(b.toString(),function(e,f){var g=B("ytPubsub2Pubsub2SkipSubKey");g&&g==d||(g=function(){if(Aq[d])try{if(f&&b instanceof yq&&b!=e)try{var h=b.h,l=f;if(!l.args||!l.version)throw Error("yt.pubsub2.Data.deserialize(): serializedData is incomplete.");try{if(!h.Va){var m=new h;h.Va=m.version}var n=h.Va}catch(E){}if(!n||l.version!=n)throw Error("yt.pubsub2.Data.deserialize(): serializedData version is incompatible.");try{n=Reflect;var q=n.construct;
var u=l.args,t=u.length;if(0<t){var A=Array(t);for(l=0;l<t;l++)A[l]=u[l];var C=A}else C=[];f=q.call(n,h,C)}catch(E){throw E.message="yt.pubsub2.Data.deserialize(): "+E.message,E;}}catch(E){throw E.message="yt.pubsub2.pubsub2 cross-binary conversion error for "+b.toString()+": "+E.message,E;}a.call(window,f)}catch(E){Pm(E)}},Cq[b.toString()]?B("yt.scheduler.instance")?Di.ea(g):kn(g,0):g())});
Aq[d]=!0;Bq[b.toString()]||(Bq[b.toString()]=[]);Bq[b.toString()].push(d);return d}
function Hq(){var a=Iq,b=Fq(function(c){a.apply(void 0,arguments);Jq(b)});
return b}
function Jq(a){var b=Eq();b&&("number"===typeof a&&(a=[a]),fb(a,function(c){b.unsubscribeByKey(c);delete Aq[c]}))}
function Eq(){return B("ytPubsub2Pubsub2Instance")}
;function Kq(a,b,c){c=void 0===c?{sampleRate:.1}:c;Math.random()<Math.min(.02,c.sampleRate/100)&&Dq("meta_logging_csi_event",{timerName:a,Ng:b})}
;var Lq=dn("max_body_size_to_compress",5E5),Mq=dn("min_body_size_to_compress",500),Nq=!0,Oq=0,Pq=0,Qq=dn("compression_performance_threshold",250),Rq=dn("slow_compressions_before_abandon_count",10);
function Sq(a,b,c,d){var e=R(),f={startTime:e},g={startTime:e,ticks:{},infos:{}};if(Nq)try{var h=Tq(b);if(!(h>Lq||h<Mq)){var l=Fk(yi(b)),m=R();f.endTime=m;g.ticks.gelc=m;Pq++;P("disable_compression_due_to_performance_degredation")&&m-e>=Qq&&(Oq++,P("abandon_compression_after_N_slow_zips")?Pq===dn("compression_disable_point")&&Oq>Rq&&(Nq=!1):Nq=!1);P("gel_compression_csi_killswitch")||!P("log_gel_compression_latency")&&!P("log_gel_compression_latency_lr")||(P("use_new_cml")?Kq("gel_compression",g,
{sampleRate:.1}):.001>=Math.random()&&Dq("gel_compression_latency_payload",f));if(Uq(l,b)||!P("only_compress_gel_if_smaller"))c.headers||(c.headers={}),c.headers["Content-Encoding"]="gzip",c.postBody=l,c.postParams=void 0}d(a,c)}catch(n){Qm(n),d(a,c)}else d(a,c)}
function Vq(a){if(!a.body)return a;try{var b="string"===typeof a.body?a.body:JSON.stringify(a.body),c=Tq(b);if(c>Lq||c<Mq)return a;var d=Fk(yi(b));if(!Uq(d,b)&&P("only_compress_gel_if_smaller"))return a;a.headers=Object.assign({},{"Content-Encoding":"gzip"},a.headers||{});a.body=d;return a}catch(e){return Qm(e),a}}
function Uq(a,b){if(!window.Blob)return!0;var c=a.length<Tq(b);c||Pm(new lo("Compressed req body is larger than uncompressed","original size: "+Tq(b),"compressed size: "+a.length));return c}
function Tq(a){return(new Blob(a.split(""))).size}
;function Wq(a){a=Object.assign({},a);delete a.Authorization;var b=Rg();if(b){var c=new Ji;c.update(O("INNERTUBE_API_KEY"));c.update(b);a.hash=od(c.digest(),3)}return a}
;var Xq;function Yq(){Xq||(Xq=new uo("yt.innertube"));return Xq}
function Zq(a,b,c,d){if(d)return null;d=Yq().get("nextId",!0)||1;var e=Yq().get("requests",!0)||{};e[d]={method:a,request:b,authState:Wq(c),requestTime:Math.round(R())};Yq().set("nextId",d+1,86400,!0);Yq().set("requests",e,86400,!0);return d}
function $q(a){var b=Yq().get("requests",!0)||{};delete b[a];Yq().set("requests",b,86400,!0)}
function ar(a){var b=Yq().get("requests",!0);if(b){for(var c in b){var d=b[c];if(!(6E4>Math.round(R())-d.requestTime)){var e=d.authState,f=Wq(wq(!1));sb(e,f)&&(e=d.request,"requestTimeMs"in e&&(e.requestTimeMs=Math.round(R())),br(a,d.method,e,{}));delete b[c]}}Yq().set("requests",b,86400,!0)}}
;function cr(a){this.Zb=this.h=!1;this.potentialEsfErrorCounter=this.i=0;this.handleError=function(){};
this.ub=function(){};
this.now=Date.now;this.Jb=!1;var b;this.Od=null!=(b=a.Od)?b:100;var c;this.Id=null!=(c=a.Id)?c:1;var d;this.Gd=null!=(d=a.Gd)?d:2592E6;var e;this.Ed=null!=(e=a.Ed)?e:12E4;var f;this.Hd=null!=(f=a.Hd)?f:5E3;var g;this.N=null!=(g=a.N)?g:void 0;this.fc=!!a.fc;var h;this.dc=null!=(h=a.dc)?h:.1;var l;this.mc=null!=(l=a.mc)?l:10;a.handleError&&(this.handleError=a.handleError);a.ub&&(this.ub=a.ub);a.Jb&&(this.Jb=a.Jb);a.Zb&&(this.Zb=a.Zb);this.O=a.O;this.ya=a.ya;this.W=a.W;this.U=a.U;this.Na=a.Na;this.Lc=
a.Lc;this.Kc=a.Kc;dr(this)&&(!this.O||this.O("networkless_logging"))&&er(this)}
function er(a){dr(a)&&!a.Jb&&(a.h=!0,a.fc&&Math.random()<=a.dc&&a.W.ge(a.N),fr(a),a.U.oa()&&a.Rb(),a.U.La(a.Lc,a.Rb.bind(a)),a.U.La(a.Kc,a.ed.bind(a)))}
k=cr.prototype;k.writeThenSend=function(a,b){var c=this;b=void 0===b?{}:b;if(dr(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.W.set(d,this.N).then(function(e){d.id=e;c.U.oa()&&gr(c,d)}).catch(function(e){gr(c,d);
hr(c,e)})}else this.Na(a,b)};
k.sendThenWrite=function(a,b,c){var d=this;b=void 0===b?{}:b;if(dr(this)&&this.h){var e={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0};this.O&&this.O("nwl_skip_retry")&&(e.skipRetry=c);if(this.U.oa()||this.O&&this.O("nwl_aggressive_send_then_write")&&!e.skipRetry){if(!e.skipRetry){var f=b.onError?b.onError:function(){};
b.onError=function(g,h){return x(function(l){if(1==l.h)return w(l,d.W.set(e,d.N).catch(function(m){hr(d,m)}),2);
f(g,h);l.h=0})}}this.Na(a,b,e.skipRetry)}else this.W.set(e,this.N).catch(function(g){d.Na(a,b,e.skipRetry);
hr(d,g)})}else this.Na(a,b,this.O&&this.O("nwl_skip_retry")&&c)};
k.sendAndWrite=function(a,b){var c=this;b=void 0===b?{}:b;if(dr(this)&&this.h){var d={url:a,options:b,timestamp:this.now(),status:"NEW",sendCount:0},e=!1,f=b.onSuccess?b.onSuccess:function(){};
d.options.onSuccess=function(g,h){void 0!==d.id?c.W.tb(d.id,c.N):e=!0;c.U.ib&&c.O&&c.O("vss_network_hint")&&c.U.ib(!0);f(g,h)};
this.Na(d.url,d.options);this.W.set(d,this.N).then(function(g){d.id=g;e&&c.W.tb(d.id,c.N)}).catch(function(g){hr(c,g)})}else this.Na(a,b)};
k.Rb=function(){var a=this;if(!dr(this))throw To("throttleSend");this.i||(this.i=this.ya.ea(function(){var b;return x(function(c){if(1==c.h)return w(c,a.W.qd("NEW",a.N),2);if(3!=c.h)return b=c.i,b?w(c,gr(a,b),3):(a.ed(),c.return());a.i&&(a.i=0,a.Rb());c.h=0})},this.Od))};
k.ed=function(){this.ya.Ba(this.i);this.i=0};
function gr(a,b){var c,d;return x(function(e){switch(e.h){case 1:if(!dr(a))throw c=To("immediateSend"),c;if(void 0===b.id){e.A(2);break}return w(e,a.W.De(b.id,a.N),3);case 3:(d=e.i)||a.ub(Error("The request cannot be found in the database."));case 2:if(ir(a,b,a.Gd)){e.A(4);break}a.ub(Error("Networkless Logging: Stored logs request expired age limit"));if(void 0===b.id){e.A(5);break}return w(e,a.W.tb(b.id,a.N),5);case 5:return e.return();case 4:b.skipRetry||(b=jr(a,b));if(!b){e.A(0);break}if(!b.skipRetry||
void 0===b.id){e.A(8);break}return w(e,a.W.tb(b.id,a.N),8);case 8:a.Na(b.url,b.options,!!b.skipRetry),e.h=0}})}
function jr(a,b){if(!dr(a))throw To("updateRequestHandlers");var c=b.options.onError?b.options.onError:function(){};
b.options.onError=function(e,f){var g,h,l,m;return x(function(n){switch(n.h){case 1:g=kr(f);(h=lr(f))&&a.O&&a.O("web_enable_error_204")&&a.handleError(Error("Request failed due to compression"),b.url,f);if(!(a.O&&a.O("nwl_consider_error_code")&&g||a.O&&!a.O("nwl_consider_error_code")&&a.potentialEsfErrorCounter<=a.mc)){n.A(2);break}if(!a.U.pc){n.A(3);break}return w(n,a.U.pc(),3);case 3:if(a.U.oa()){n.A(2);break}c(e,f);if(!a.O||!a.O("nwl_consider_error_code")||void 0===(null==(l=b)?void 0:l.id)){n.A(6);
break}return w(n,a.W.Pc(b.id,a.N,!1),6);case 6:return n.return();case 2:if(a.O&&a.O("nwl_consider_error_code")&&!g&&a.potentialEsfErrorCounter>a.mc)return n.return();a.potentialEsfErrorCounter++;if(void 0===(null==(m=b)?void 0:m.id)){n.A(8);break}return b.sendCount<a.Id?w(n,a.W.Pc(b.id,a.N,!0,h?!1:void 0),12):w(n,a.W.tb(b.id,a.N),8);case 12:a.ya.ea(function(){a.U.oa()&&a.Rb()},a.Hd);
case 8:c(e,f),n.h=0}})};
var d=b.options.onSuccess?b.options.onSuccess:function(){};
b.options.onSuccess=function(e,f){var g;return x(function(h){if(1==h.h)return void 0===(null==(g=b)?void 0:g.id)?h.A(2):w(h,a.W.tb(b.id,a.N),2);a.U.ib&&a.O&&a.O("vss_network_hint")&&a.U.ib(!0);d(e,f);h.h=0})};
return b}
function ir(a,b,c){b=b.timestamp;return a.now()-b>=c?!1:!0}
function fr(a){if(!dr(a))throw To("retryQueuedRequests");a.W.qd("QUEUED",a.N).then(function(b){b&&!ir(a,b,a.Ed)?a.ya.ea(function(){return x(function(c){if(1==c.h)return void 0===b.id?c.A(2):w(c,a.W.Pc(b.id,a.N),2);fr(a);c.h=0})}):a.U.oa()&&a.Rb()})}
function hr(a,b){a.Ud&&!a.U.oa()?a.Ud(b):a.handleError(b)}
function dr(a){return!!a.N||a.Zb}
function kr(a){var b;return(a=null==a?void 0:null==(b=a.error)?void 0:b.code)&&400<=a&&599>=a?!1:!0}
function lr(a){var b;a=null==a?void 0:null==(b=a.error)?void 0:b.code;return!(400!==a&&415!==a)}
;var mr;
function nr(){if(mr)return mr();var a={};mr=bq("LogsDatabaseV2",{Ab:(a.LogsRequestsStore={Gb:2},a),qc:!1,upgrade:function(b,c,d){c(2)&&hp(b,"LogsRequestsStore",{keyPath:"id",autoIncrement:!0});c(3);c(5)&&(d=d.objectStore("LogsRequestsStore"),d.h.indexNames.contains("newRequest")&&d.h.deleteIndex("newRequest"),op(d,"newRequestV2",["status","interface","timestamp"]));c(7)&&b.h.objectStoreNames.contains("sapisid")&&b.h.deleteObjectStore("sapisid");c(9)&&b.h.objectStoreNames.contains("SWHealthLog")&&b.h.deleteObjectStore("SWHealthLog")},
version:9});return mr()}
;function or(a){return Ap(nr(),a)}
function pr(a,b){var c,d,e,f,g;return x(function(h){if(1==h.h)return c={startTime:R(),transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},d={startTime:R(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_WRITE"},ticks:{}},w(h,or(b),2);if(3!=h.h)return e=h.i,f=Object.assign({},a,{options:JSON.parse(JSON.stringify(a.options)),interface:O("INNERTUBE_CONTEXT_CLIENT_NAME",0)}),w(h,jp(e,"LogsRequestsStore",f),3);g=h.i;c.kf=R();d.ticks.tc=R();qr(d,c);return h.return(g)})}
function rr(a,b){var c,d,e,f,g,h,l,m;return x(function(n){if(1==n.h)return c={startTime:R(),transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},d={startTime:R(),infos:{transactionType:"YT_IDB_TRANSACTION_TYPE_READ"},ticks:{}},w(n,or(b),2);if(3!=n.h)return e=n.i,f=O("INNERTUBE_CONTEXT_CLIENT_NAME",0),g=[a,f,0],h=[a,f,R()],l=IDBKeyRange.bound(g,h),m=void 0,w(n,gp(e,["LogsRequestsStore"],{mode:"readwrite",ca:!0},function(q){return tp(q.objectStore("LogsRequestsStore").index("newRequestV2"),{query:l,direction:"prev"},
function(u){u.getValue()&&(m=u.getValue(),"NEW"===a&&(m.status="QUEUED",u.update(m)))})}),3);
c.kf=R();d.ticks.tc=R();qr(d,c);return n.return(m)})}
function Ar(a,b){var c;return x(function(d){if(1==d.h)return w(d,or(b),2);c=d.i;return d.return(gp(c,["LogsRequestsStore"],{mode:"readwrite",ca:!0},function(e){var f=e.objectStore("LogsRequestsStore");return f.get(a).then(function(g){if(g)return g.status="QUEUED",cp(f.h.put(g,void 0)).then(function(){return g})})}))})}
function Er(a,b,c,d){c=void 0===c?!0:c;var e;return x(function(f){if(1==f.h)return w(f,or(b),2);e=f.i;return f.return(gp(e,["LogsRequestsStore"],{mode:"readwrite",ca:!0},function(g){var h=g.objectStore("LogsRequestsStore");return h.get(a).then(function(l){return l?(l.status="NEW",c&&(l.sendCount+=1),void 0!==d&&(l.options.compress=d),cp(h.h.put(l,void 0)).then(function(){return l})):Xo.resolve(void 0)})}))})}
function Xr(a,b){var c;return x(function(d){if(1==d.h)return w(d,or(b),2);c=d.i;return d.return(c.delete("LogsRequestsStore",a))})}
function Yr(a){var b,c;return x(function(d){if(1==d.h)return w(d,or(a),2);b=d.i;c=R()-2592E6;return w(d,gp(b,["LogsRequestsStore"],{mode:"readwrite",ca:!0},function(e){return qp(e.objectStore("LogsRequestsStore"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return f.continue()})})}),0)})}
function Zr(){x(function(a){return w(a,Yp(),0)})}
function qr(a,b){P("nwl_csi_killswitch")||(P("use_new_cml")?Kq("networkless_performance",a,{sampleRate:1}):.01>=Math.random()&&Dq("nwl_transaction_latency_payload",b))}
;var $r={},as=bq("ServiceWorkerLogsDatabase",{Ab:($r.SWHealthLog={Gb:1},$r),qc:!0,upgrade:function(a,b){b(1)&&op(hp(a,"SWHealthLog",{keyPath:"id",autoIncrement:!0}),"swHealthNewRequest",["interface","timestamp"])},
version:1});function bs(a){return Ap(as(),a)}
function cs(a){var b,c;x(function(d){if(1==d.h)return w(d,bs(a),2);b=d.i;c=R()-2592E6;return w(d,gp(b,["SWHealthLog"],{mode:"readwrite",ca:!0},function(e){return qp(e.objectStore("SWHealthLog"),{},function(f){if(f.getValue().timestamp<=c)return f.delete().then(function(){return f.continue()})})}),0)})}
function ds(a){var b;return x(function(c){if(1==c.h)return w(c,bs(a),2);b=c.i;return w(c,b.clear("SWHealthLog"),0)})}
;var es={},fs=0;function gs(a){var b=new Image,c=""+fs++;es[c]=b;b.onload=b.onerror=function(){delete es[c]};
b.src=a}
;function hs(){this.h=new Map;this.i=!1}
function is(){if(!hs.h){var a=B("yt.networkRequestMonitor.instance")||new hs;z("yt.networkRequestMonitor.instance",a);hs.h=a}return hs.h}
hs.prototype.requestComplete=function(a,b){b&&(this.i=!0);a=this.removeParams(a);this.h.get(a)||this.h.set(a,b)};
hs.prototype.isEndpointCFR=function(a){a=this.removeParams(a);return(a=this.h.get(a))?!1:!1===a&&this.i?!0:null};
hs.prototype.removeParams=function(a){return a.split("?")[0]};
hs.prototype.removeParams=hs.prototype.removeParams;hs.prototype.isEndpointCFR=hs.prototype.isEndpointCFR;hs.prototype.requestComplete=hs.prototype.requestComplete;hs.getInstance=is;var js;function ks(){js||(js=new uo("yt.offline"));return js}
function ls(a){if(P("offline_error_handling")){var b=ks().get("errors",!0)||{};b[a.message]={name:a.name,stack:a.stack};a.level&&(b[a.message].level=a.level);ks().set("errors",b,2592E3,!0)}}
;function ms(){tf.call(this);var a=this;this.j=!1;this.i=Ci();this.i.La("networkstatus-online",function(){if(a.j&&P("offline_error_handling")){var b=ks().get("errors",!0);if(b){for(var c in b)if(b[c]){var d=new lo(c,"sent via offline_errors");d.name=b[c].name;d.stack=b[c].stack;d.level=b[c].level;Pm(d)}ks().set("errors",{},2592E3,!0)}}})}
v(ms,tf);function ns(){if(!ms.h){var a=B("yt.networkStatusManager.instance")||new ms;z("yt.networkStatusManager.instance",a);ms.h=a}return ms.h}
k=ms.prototype;k.oa=function(){return this.i.oa()};
k.ib=function(a){this.i.i=a};
k.te=function(){var a=window.navigator.onLine;return void 0===a?!0:a};
k.le=function(){this.j=!0};
k.La=function(a,b){return this.i.La(a,b)};
k.pc=function(a){a=Ai(this.i,a);a.then(function(b){P("use_cfr_monitor")&&is().requestComplete("generate_204",b)});
return a};
ms.prototype.sendNetworkCheckRequest=ms.prototype.pc;ms.prototype.listen=ms.prototype.La;ms.prototype.enableErrorFlushing=ms.prototype.le;ms.prototype.getWindowStatus=ms.prototype.te;ms.prototype.networkStatusHint=ms.prototype.ib;ms.prototype.isNetworkAvailable=ms.prototype.oa;ms.getInstance=ns;function os(a){a=void 0===a?{}:a;tf.call(this);var b=this;this.i=this.s=0;this.j=ns();var c=B("yt.networkStatusManager.instance.listen").bind(this.j);c&&(a.oc?(this.oc=a.oc,c("networkstatus-online",function(){ps(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){ps(b,"publicytnetworkstatus-offline")})):(c("networkstatus-online",function(){uf(b,"publicytnetworkstatus-online")}),c("networkstatus-offline",function(){uf(b,"publicytnetworkstatus-offline")})))}
v(os,tf);os.prototype.oa=function(){var a=B("yt.networkStatusManager.instance.isNetworkAvailable");return a?a.bind(this.j)():!0};
os.prototype.ib=function(a){var b=B("yt.networkStatusManager.instance.networkStatusHint").bind(this.j);b&&b(a)};
os.prototype.pc=function(a){var b=this,c;return x(function(d){c=B("yt.networkStatusManager.instance.sendNetworkCheckRequest").bind(b.j);return P("skip_network_check_if_cfr")&&is().isEndpointCFR("generate_204")?d.return(new Promise(function(e){var f;b.ib((null==(f=window.navigator)?void 0:f.onLine)||!0);e(b.oa())})):c?d.return(c(a)):d.return(!0)})};
function ps(a,b){a.oc?a.i?(Di.Ba(a.s),a.s=Di.ea(function(){a.m!==b&&(uf(a,b),a.m=b,a.i=R())},a.oc-(R()-a.i))):(uf(a,b),a.m=b,a.i=R()):uf(a,b)}
;var qs;function rs(){var a=cr.call;qs||(qs=new os({Ag:!0,ug:!0}));a.call(cr,this,{W:{ge:Yr,tb:Xr,qd:rr,De:Ar,Pc:Er,set:pr},U:qs,handleError:function(b,c,d){var e,f=null==d?void 0:null==(e=d.error)?void 0:e.code;if(400===f||415===f){var g;Pm(new lo(b.message,c,null==d?void 0:null==(g=d.error)?void 0:g.code),void 0,void 0,void 0,!0)}else Pm(b)},
ub:Qm,Na:ss,now:R,Ud:ls,ya:to(),Lc:"publicytnetworkstatus-online",Kc:"publicytnetworkstatus-offline",fc:!0,dc:.1,mc:dn("potential_esf_error_limit",10),O:P,Jb:!(mo()&&ts())});this.j=new $h;P("networkless_immediately_drop_all_requests")&&Zr();Zp("LogsDatabaseV2")}
v(rs,cr);function us(){var a=B("yt.networklessRequestController.instance");a||(a=new rs,z("yt.networklessRequestController.instance",a),P("networkless_logging")&&Op().then(function(b){a.N=b;er(a);a.j.resolve();a.fc&&Math.random()<=a.dc&&a.N&&cs(a.N);P("networkless_immediately_drop_sw_health_store")&&vs(a)}));
return a}
rs.prototype.writeThenSend=function(a,b){b||(b={});mo()||(this.h=!1);cr.prototype.writeThenSend.call(this,a,b)};
rs.prototype.sendThenWrite=function(a,b,c){b||(b={});mo()||(this.h=!1);cr.prototype.sendThenWrite.call(this,a,b,c)};
rs.prototype.sendAndWrite=function(a,b){b||(b={});mo()||(this.h=!1);cr.prototype.sendAndWrite.call(this,a,b)};
rs.prototype.awaitInitialization=function(){return this.j.promise};
function vs(a){var b;x(function(c){if(!a.N)throw b=To("clearSWHealthLogsDb"),b;return c.return(ds(a.N).catch(function(d){a.handleError(d)}))})}
function ss(a,b,c){P("use_cfr_monitor")&&ws(a,b);if(P("use_request_time_ms_header"))b.headers&&(b.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(R())));else{var d;if(null==(d=b.postParams)?0:d.requestTimeMs)b.postParams.requestTimeMs=Math.round(R())}if(c&&0===Object.keys(b).length){var e=void 0===e?"":e;var f=void 0===f?!1:f;if(a)if(e)tn(a,void 0,"POST",e);else if(O("USE_NET_AJAX_FOR_PING_TRANSPORT",!1))tn(a,void 0,"GET","",void 0,void 0,f);else{b:{try{var g=new ab({url:a});if(g.j&&g.i||
g.l){var h=uc(vc(5,a)),l;if(!(l=!h||!h.endsWith("/aclk"))){var m=a.search(Gc),n=Fc(a,0,"ri",m);if(0>n)var q=null;else{var u=a.indexOf("&",n);if(0>u||u>m)u=m;q=decodeURIComponent(a.slice(n+3,-1!==u?u:0).replace(/\+/g," "))}l="1"!==q}var t=!l;break b}}catch(C){}t=!1}if(t){b:{try{if(window.navigator&&window.navigator.sendBeacon&&window.navigator.sendBeacon(a,"")){var A=!0;break b}}catch(C){}A=!1}c=A?!0:!1}else c=!1;c||gs(a)}}else b.compress?b.postBody?("string"!==typeof b.postBody&&(b.postBody=JSON.stringify(b.postBody)),
Sq(a,b.postBody,b,qn)):Sq(a,JSON.stringify(b.postParams),b,yn):qn(a,b)}
function ws(a,b){var c=b.onError?b.onError:function(){};
b.onError=function(e,f){is().requestComplete(a,!1);c(e,f)};
var d=b.onSuccess?b.onSuccess:function(){};
b.onSuccess=function(e,f){is().requestComplete(a,!0);d(e,f)}}
function ts(){return"www.youtube-nocookie.com"!==wc(document.location.toString())}
;var xs=!1,ys=y.ytNetworklessLoggingInitializationOptions||{isNwlInitialized:xs};z("ytNetworklessLoggingInitializationOptions",ys);function zs(){var a;x(function(b){if(1==b.h)return w(b,Op(),2);a=b.i;if(!a||!mo()&&!P("nwl_init_require_datasync_id_killswitch")||!ts())return b.A(0);xs=!0;ys.isNwlInitialized=xs;return w(b,us().awaitInitialization(),0)})}
;function As(a){var b=this;this.config_=null;a?this.config_=a:mq()&&(this.config_=nq());po(function(){ar(b)},5E3)}
As.prototype.isReady=function(){!this.config_&&mq()&&(this.config_=nq());return!!this.config_};
function br(a,b,c,d){function e(A){A=void 0===A?!1:A;var C;if(d.retry&&"www.youtube-nocookie.com"!=h&&(A||P("skip_ls_gel_retry")||"application/json"!==g.headers["Content-Type"]||(C=Zq(b,c,m,l)),C)){var E=g.onSuccess,M=g.onFetchSuccess;g.onSuccess=function(S,V){$q(C);E(S,V)};
c.onFetchSuccess=function(S,V){$q(C);M(S,V)}}try{if(A&&d.retry&&!d.zd.bypassNetworkless)g.method="POST",d.zd.writeThenSend?us().writeThenSend(t,g):us().sendAndWrite(t,g);
else if(d.compress)if(g.postBody){var Q=g.postBody;"string"!==typeof Q&&(Q=JSON.stringify(g.postBody));Sq(t,Q,g,qn)}else Sq(t,JSON.stringify(g.postParams),g,yn);else P("web_all_payloads_via_jspb")?qn(t,g):yn(t,g)}catch(S){if("InvalidAccessError"==S.name)C&&($q(C),C=0),Qm(Error("An extension is blocking network request."));else throw S;}C&&po(function(){ar(a)},5E3)}
!O("VISITOR_DATA")&&"visitor_id"!==b&&.01>Math.random()&&Qm(new lo("Missing VISITOR_DATA when sending innertube request.",b,c,d));if(!a.isReady()){var f=new lo("innertube xhrclient not ready",b,c,d);Pm(f);throw f;}var g={headers:d.headers||{},method:"POST",postParams:c,postBody:d.postBody,postBodyFormat:d.postBodyFormat||"JSON",onTimeout:function(){d.onTimeout()},
onFetchTimeout:d.onTimeout,onSuccess:function(A,C){if(d.onSuccess)d.onSuccess(C)},
onFetchSuccess:function(A){if(d.onSuccess)d.onSuccess(A)},
onError:function(A,C){if(d.onError)d.onError(C)},
onFetchError:function(A){if(d.onError)d.onError(A)},
timeout:d.timeout,withCredentials:!0,compress:d.compress};g.headers["Content-Type"]||(g.headers["Content-Type"]="application/json");var h="";(f=a.config_.ze)&&(h=f);var l=a.config_.Be||!1,m=wq(l,h,d);Object.assign(g.headers,m);(f=g.headers.Authorization)&&!h&&l&&(g.headers["x-origin"]=window.location.origin);var n="/youtubei/"+a.config_.innertubeApiVersion+"/"+b,q={alt:"json"},u=a.config_.Ae&&f;u=u&&f.startsWith("Bearer");u||(q.key=a.config_.innertubeApiKey);var t=Zm(""+h+n,q||{},!0);(B("ytNetworklessLoggingInitializationOptions")?
ys.isNwlInitialized:xs)?Mp().then(function(A){e(A)}):e(!1)}
;var Bs=0,Cs=Zc?"webkit":Yc?"moz":Wc?"ms":Vc?"o":"";z("ytDomDomGetNextId",B("ytDomDomGetNextId")||function(){return++Bs});var Ds={stopImmediatePropagation:1,stopPropagation:1,preventMouseEvent:1,preventManipulation:1,preventDefault:1,layerX:1,layerY:1,screenX:1,screenY:1,scale:1,rotation:1,webkitMovementX:1,webkitMovementY:1};
function Es(a){this.type="";this.state=this.source=this.data=this.currentTarget=this.relatedTarget=this.target=null;this.charCode=this.keyCode=0;this.metaKey=this.shiftKey=this.ctrlKey=this.altKey=!1;this.rotation=this.clientY=this.clientX=0;this.scale=1;this.changedTouches=this.touches=null;try{if(a=a||window.event){this.event=a;for(var b in a)b in Ds||(this[b]=a[b]);this.scale=a.scale;this.rotation=a.rotation;var c=a.target||a.srcElement;c&&3==c.nodeType&&(c=c.parentNode);this.target=c;var d=a.relatedTarget;
if(d)try{d=d.nodeName?d:null}catch(e){d=null}else"mouseover"==this.type?d=a.fromElement:"mouseout"==this.type&&(d=a.toElement);this.relatedTarget=d;this.clientX=void 0!=a.clientX?a.clientX:a.pageX;this.clientY=void 0!=a.clientY?a.clientY:a.pageY;this.keyCode=a.keyCode?a.keyCode:a.which;this.charCode=a.charCode||("keypress"==this.type?this.keyCode:0);this.altKey=a.altKey;this.ctrlKey=a.ctrlKey;this.shiftKey=a.shiftKey;this.metaKey=a.metaKey;this.h=a.pageX;this.i=a.pageY}}catch(e){}}
function Fs(a){if(document.body&&document.documentElement){var b=document.body.scrollTop+document.documentElement.scrollTop;a.h=a.clientX+(document.body.scrollLeft+document.documentElement.scrollLeft);a.i=a.clientY+b}}
Es.prototype.preventDefault=function(){this.event&&(this.event.returnValue=!1,this.event.preventDefault&&this.event.preventDefault())};
Es.prototype.stopPropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopPropagation&&this.event.stopPropagation())};
Es.prototype.stopImmediatePropagation=function(){this.event&&(this.event.cancelBubble=!0,this.event.stopImmediatePropagation&&this.event.stopImmediatePropagation())};var ob=y.ytEventsEventsListeners||{};z("ytEventsEventsListeners",ob);var Gs=y.ytEventsEventsCounter||{count:0};z("ytEventsEventsCounter",Gs);
function Hs(a,b,c,d){d=void 0===d?{}:d;a.addEventListener&&("mouseenter"!=b||"onmouseenter"in document?"mouseleave"!=b||"onmouseenter"in document?"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"):b="mouseout":b="mouseover");return nb(function(e){var f="boolean"===typeof e[4]&&e[4]==!!d,g=Pa(e[4])&&Pa(d)&&sb(e[4],d);return!!e.length&&e[0]==a&&e[1]==b&&e[2]==c&&(f||g)})}
var Is=db(function(){var a=!1;try{var b=Object.defineProperty({},"capture",{get:function(){a=!0}});
window.addEventListener("test",null,b)}catch(c){}return a});
function Js(a,b,c,d){d=void 0===d?{}:d;if(!a||!a.addEventListener&&!a.attachEvent)return"";var e=Hs(a,b,c,d);if(e)return e;e=++Gs.count+"";var f=!("mouseenter"!=b&&"mouseleave"!=b||!a.addEventListener||"onmouseenter"in document);var g=f?function(h){h=new Es(h);if(!Df(h.relatedTarget,function(l){return l==a}))return h.currentTarget=a,h.type=b,c.call(a,h)}:function(h){h=new Es(h);
h.currentTarget=a;return c.call(a,h)};
g=Om(g);a.addEventListener?("mouseenter"==b&&f?b="mouseover":"mouseleave"==b&&f?b="mouseout":"mousewheel"==b&&"MozBoxSizing"in document.documentElement.style&&(b="MozMousePixelScroll"),Is()||"boolean"===typeof d?a.addEventListener(b,g,d):a.addEventListener(b,g,!!d.capture)):a.attachEvent("on"+b,g);ob[e]=[a,b,c,g,d];return e}
function Ks(a){a&&("string"==typeof a&&(a=[a]),fb(a,function(b){if(b in ob){var c=ob[b],d=c[0],e=c[1],f=c[3];c=c[4];d.removeEventListener?Is()||"boolean"===typeof c?d.removeEventListener(e,f,c):d.removeEventListener(e,f,!!c.capture):d.detachEvent&&d.detachEvent("on"+e,f);delete ob[b]}}))}
;function Ls(a){this.H=a;this.i=null;this.m=0;this.v=null;this.s=0;this.j=[];for(a=0;4>a;a++)this.j.push(0);this.l=0;this.T=Js(window,"mousemove",Wa(this.V,this));a=Wa(this.M,this);"function"===typeof a&&(a=Om(a));this.Y=window.setInterval(a,25)}
Ya(Ls,K);Ls.prototype.V=function(a){void 0===a.h&&Fs(a);var b=a.h;void 0===a.i&&Fs(a);this.i=new zf(b,a.i)};
Ls.prototype.M=function(){if(this.i){var a=R();if(0!=this.m){var b=this.v,c=this.i,d=b.x-c.x;b=b.y-c.y;d=Math.sqrt(d*d+b*b)/(a-this.m);this.j[this.l]=.5<Math.abs((d-this.s)/this.s)?1:0;for(c=b=0;4>c;c++)b+=this.j[c]||0;3<=b&&this.H();this.s=d}this.m=a;this.v=this.i;this.l=(this.l+1)%4}};
Ls.prototype.L=function(){window.clearInterval(this.Y);Ks(this.T)};var Ms={};
function Ns(a){var b=void 0===a?{}:a;a=void 0===b.Me?!1:b.Me;b=void 0===b.me?!0:b.me;if(null==B("_lact",window)){var c=parseInt(O("LACT"),10);c=isFinite(c)?Date.now()-Math.max(c,0):-1;z("_lact",c,window);z("_fact",c,window);-1==c&&Os();Js(document,"keydown",Os);Js(document,"keyup",Os);Js(document,"mousedown",Os);Js(document,"mouseup",Os);a?Js(window,"touchmove",function(){Ps("touchmove",200)},{passive:!0}):(Js(window,"resize",function(){Ps("resize",200)}),b&&Js(window,"scroll",function(){Ps("scroll",200)}));
new Ls(function(){Ps("mouse",100)});
Js(document,"touchstart",Os,{passive:!0});Js(document,"touchend",Os,{passive:!0})}}
function Ps(a,b){Ms[a]||(Ms[a]=!0,Di.ea(function(){Os();Ms[a]=!1},b))}
function Os(){null==B("_lact",window)&&Ns();var a=Date.now();z("_lact",a,window);-1==B("_fact",window)&&z("_fact",a,window);(a=B("ytglobal.ytUtilActivityCallback_"))&&a()}
function Qs(){var a=B("_lact",window);return null==a?-1:Math.max(Date.now()-a,0)}
;var Rs=y.ytPubsubPubsubInstance||new $i,Ss=y.ytPubsubPubsubSubscribedKeys||{},Ts=y.ytPubsubPubsubTopicToKeys||{},Us=y.ytPubsubPubsubIsSynchronous||{};function Vs(a,b){var c=Ws();if(c&&b){var d=c.subscribe(a,function(){var e=arguments;var f=function(){Ss[d]&&b.apply&&"function"==typeof b.apply&&b.apply(window,e)};
try{Us[a]?f():kn(f,0)}catch(g){Pm(g)}},void 0);
Ss[d]=!0;Ts[a]||(Ts[a]=[]);Ts[a].push(d);return d}return 0}
function Xs(a){var b=Ws();b&&("number"===typeof a?a=[a]:"string"===typeof a&&(a=[parseInt(a,10)]),fb(a,function(c){b.unsubscribeByKey(c);delete Ss[c]}))}
function Ys(a,b){var c=Ws();c&&c.publish.apply(c,arguments)}
function Zs(a){var b=Ws();if(b)if(b.clear(a),a)$s(a);else for(var c in Ts)$s(c)}
function Ws(){return y.ytPubsubPubsubInstance}
function $s(a){Ts[a]&&(a=Ts[a],fb(a,function(b){Ss[b]&&delete Ss[b]}),a.length=0)}
$i.prototype.subscribe=$i.prototype.subscribe;$i.prototype.unsubscribeByKey=$i.prototype.Fb;$i.prototype.publish=$i.prototype.cb;$i.prototype.clear=$i.prototype.clear;z("ytPubsubPubsubInstance",Rs);z("ytPubsubPubsubTopicToKeys",Ts);z("ytPubsubPubsubIsSynchronous",Us);z("ytPubsubPubsubSubscribedKeys",Ss);var at=Symbol("injectionDeps");function bt(a){this.name=a}
bt.prototype.toString=function(){return"InjectionToken("+this.name+")"};
function ct(a){this.key=a}
function dt(){this.h=new Map;this.i=new Map}
dt.prototype.resolve=function(a){return a instanceof ct?et(this,a.key,[],!0):et(this,a,[])};
function et(a,b,c,d){d=void 0===d?!1:d;if(-1<c.indexOf(b))throw Error("Deps cycle for: "+b);if(a.i.has(b))return a.i.get(b);if(!a.h.has(b)){if(d)return;throw Error("No provider for: "+b);}d=a.h.get(b);c.push(b);if(d.Rd)var e=d.Rd;else if(d.nf)e=d[at]?ft(a,d[at],c):[],e=d.nf.apply(d,ha(e));else if(d.Qd){e=d.Qd;var f=e[at]?ft(a,e[at],c):[];e=new (Function.prototype.bind.apply(e,[null].concat(ha(f))))}else throw Error("Could not resolve providers for: "+b);c.pop();d.Lg||a.i.set(b,e);return e}
function ft(a,b,c){return b?b.map(function(d){return d instanceof ct?et(a,d.key,c,!0):et(a,d,c)}):[]}
;var gt;function ht(){gt||(gt=new dt);return gt}
;function jt(){this.store={};this.h={}}
jt.prototype.storePayload=function(a,b){a=kt(a);this.store[a]?this.store[a].push(b):(this.h={},this.store[a]=[b]);return a};
jt.prototype.extractMatchingEntries=function(a){a=lt(this,a);for(var b=[],c=0;c<a.length;c++)this.store[a[c]]&&(b.push.apply(b,ha(this.store[a[c]])),delete this.store[a[c]]);return b};
jt.prototype.getSequenceCount=function(a){a=lt(this,a);for(var b=0,c=0;c<a.length;c++)b+=this.store[a[c]].length||0;return b};
function lt(a,b){var c=kt(b);if(a.h[c])return a.h[c];var d=Object.keys(a.store)||[];if(1>=d.length&&kt(b)===d[0])return d;for(var e=[],f=0;f<d.length;f++){var g=d[f].split("/");if(mt(b.auth,g[0])){var h=b.isJspb;mt(void 0===h?"undefined":h?"true":"false",g[1])&&mt(b.cttAuthInfo,g[2])&&e.push(d[f])}}return a.h[c]=e}
function mt(a,b){return void 0===a||"undefined"===a?!0:a===b}
jt.prototype.getSequenceCount=jt.prototype.getSequenceCount;jt.prototype.extractMatchingEntries=jt.prototype.extractMatchingEntries;jt.prototype.storePayload=jt.prototype.storePayload;function kt(a){return[void 0===a.auth?"undefined":a.auth,void 0===a.isJspb?"undefined":a.isJspb,void 0===a.cttAuthInfo?"undefined":a.cttAuthInfo].join("/")}
;function nt(a,b){if(a)return a[b.name]}
;var ot=dn("initial_gel_batch_timeout",2E3),pt=dn("gel_queue_timeout_max_ms",6E4),qt=Math.pow(2,16)-1,rt=void 0;function st(){this.j=this.h=this.i=0}
var tt=new st,ut=new st,vt,wt=!0,xt=y.ytLoggingTransportTokensToCttTargetIds_||{};z("ytLoggingTransportTokensToCttTargetIds_",xt);var zt=y.ytLoggingTransportTokensToJspbCttTargetIds_||{};z("ytLoggingTransportTokensToJspbCttTargetIds_",zt);var At={};function Bt(){var a=B("yt.logging.ims");a||(a=new jt,z("yt.logging.ims",a));return a}
function Ct(a,b){P("web_all_payloads_via_jspb")&&Qm(new lo("transport.log called for JSON in JSPB only experiment"));if("log_event"===a.endpoint){Dt(a);var c=Et(a);At[c]=!0;var d={cttAuthInfo:c,isJspb:!1};Bt().storePayload(d,a.payload);Ft(b,c,!1,d)}}
function Gt(a,b){if("log_event"===a.endpoint){Dt(void 0,a);var c=Et(a,!0);At[c]=!0;var d={cttAuthInfo:c,isJspb:!0};Bt().storePayload(d,a.payload.toJSON());Ft(b,c,!0,d)}}
function Ft(a,b,c,d){c=void 0===c?!1:c;a&&(rt=new a);a=dn("tvhtml5_logging_max_batch_ads_fork")||dn("tvhtml5_logging_max_batch")||dn("web_logging_max_batch")||100;var e=R(),f=c?ut.j:tt.j,g=0;d&&(g=Bt().getSequenceCount(d));g>=a?vt||(vt=Ht(function(){It({writeThenSend:!0},P("flush_only_full_queue")?b:void 0,c);vt=void 0},0)):10<=e-f&&(Jt(c),c?ut.j=e:tt.j=e)}
function Kt(a,b){P("web_all_payloads_via_jspb")&&Qm(new lo("transport.logIsolatedGelPayload called in JSPB only experiment"));if("log_event"===a.endpoint){Dt(a);var c=Et(a),d=new Map;d.set(c,[a.payload]);b&&(rt=new b);return new Pf(function(e,f){rt&&rt.isReady()?Lt(d,rt,e,f,{bypassNetworkless:!0},!0):e()})}}
function Mt(a,b){if("log_event"===a.endpoint){Dt(void 0,a);var c=Et(a,!0),d=new Map;d.set(c,[a.payload.toJSON()]);b&&(rt=new b);return new Pf(function(e){rt&&rt.isReady()?Nt(d,rt,e,{bypassNetworkless:!0},!0):e()})}}
function Et(a,b){var c="";if(a.dangerousLogToVisitorSession)c="visitorOnlyApprovedKey";else if(a.cttAuthInfo){if(void 0===b?0:b){b=a.cttAuthInfo.token;c=a.cttAuthInfo;var d=new tm;c.videoId?d.setVideoId(c.videoId):c.playlistId&&ee(d,2,um,c.playlistId);zt[b]=d}else b=a.cttAuthInfo,c={},b.videoId?c.videoId=b.videoId:b.playlistId&&(c.playlistId=b.playlistId),xt[a.cttAuthInfo.token]=c;c=a.cttAuthInfo.token}return c}
function It(a,b,c){a=void 0===a?{}:a;c=void 0===c?!1:c;!c&&P("web_all_payloads_via_jspb")&&Qm(new lo("transport.flushLogs called for JSON in JSPB only experiment"));new Pf(function(d,e){c?(Ot(ut.i),Ot(ut.h),ut.h=0):(Ot(tt.i),Ot(tt.h),tt.h=0);if(rt&&rt.isReady()){var f=a,g=c,h=rt;f=void 0===f?{}:f;g=void 0===g?!1:g;var l=new Map,m=new Map;if(void 0!==b)g?(e=Bt().extractMatchingEntries({isJspb:g,cttAuthInfo:b}),l.set(b,e),Nt(l,h,d,f)):(l=Bt().extractMatchingEntries({isJspb:g,cttAuthInfo:b}),m.set(b,
l),Lt(m,h,d,e,f));else if(g){e=r(Object.keys(At));for(g=e.next();!g.done;g=e.next())m=g.value,g=Bt().extractMatchingEntries({isJspb:!0,cttAuthInfo:m}),0<g.length&&l.set(m,g),delete At[m];Nt(l,h,d,f)}else{l=r(Object.keys(At));for(g=l.next();!g.done;g=l.next()){g=g.value;var n=Bt().extractMatchingEntries({isJspb:!1,cttAuthInfo:g});0<n.length&&m.set(g,n);delete At[g]}Lt(m,h,d,e,f)}}else Jt(c),d()})}
function Jt(a){a=void 0===a?!1:a;if(P("web_gel_timeout_cap")&&(!a&&!tt.h||a&&!ut.h)){var b=Ht(function(){It({writeThenSend:!0},void 0,a)},pt);
a?ut.h=b:tt.h=b}Ot(a?ut.i:tt.i);b=O("LOGGING_BATCH_TIMEOUT",dn("web_gel_debounce_ms",1E4));P("shorten_initial_gel_batch_timeout")&&wt&&(b=ot);b=Ht(function(){It({writeThenSend:!0},void 0,a)},b);
a?ut.i=b:tt.i=b}
function Lt(a,b,c,d,e,f){e=void 0===e?{}:e;var g=Math.round(R()),h=a.size,l={};a=r(a);for(var m=a.next();!m.done;l={Sb:l.Sb,Ya:l.Ya,Db:l.Db,Ub:l.Ub,Tb:l.Tb},m=a.next()){var n=r(m.value);m=n.next().value;n=n.next().value;l.Ya=ub({context:oq(b.config_||nq())});if(!Oa(n)&&!P("throw_err_when_logevent_malformed_killswitch")){d();break}l.Ya.events=n;(n=xt[m])&&Pt(l.Ya,m,n);delete xt[m];l.Db="visitorOnlyApprovedKey"===m;Qt(l.Ya,g,l.Db);Rt(e);l.Ub=function(q){P("update_log_event_config")&&Di.ea(function(){return x(function(u){return w(u,
St(q),0)})});
h--;h||c()};
l.Sb=0;l.Tb=function(q){return function(){q.Sb++;if(e.bypassNetworkless&&1===q.Sb)try{br(b,"log_event",q.Ya,Tt({writeThenSend:!0},q.Db,q.Ub,q.Tb,f)),wt=!1}catch(u){Pm(u),d()}h--;h||c()}}(l);
try{br(b,"log_event",l.Ya,Tt(e,l.Db,l.Ub,l.Tb,f)),wt=!1}catch(q){Pm(q),d()}}}
function Nt(a,b,c,d,e){d=void 0===d?{}:d;var f=Math.round(R()),g=a.size,h=new Map([].concat(ha(a)));h=r(h);for(var l=h.next();!l.done;l=h.next()){var m=r(l.value).next().value,n=a.get(m);l=new vm;var q=vq(b.config_||nq());H(l,dl,1,q);n=n?Ut(n):[];n=r(n);for(q=n.next();!q.done;q=n.next())le(l,3,pm,q.value);(n=zt[m])&&Vt(l,m,n);delete zt[m];m="visitorOnlyApprovedKey"===m;Wt(l,f,m);Rt(d);l=Ce(l);m=Tt(d,m,function(u){P("update_log_event_config")&&Di.ea(function(){return x(function(t){return w(t,St(u),
0)})});
g--;g||c()},function(){g--;
g||c()},e);
m.headers["Content-Type"]="application/json+protobuf";m.postBodyFormat="JSPB";m.postBody=l;br(b,"log_event","",m);wt=!1}}
function Rt(a){P("always_send_and_write")&&(a.writeThenSend=!1)}
function Tt(a,b,c,d,e){a={retry:!0,onSuccess:c,onError:d,zd:a,dangerousLogToVisitorSession:b,qg:!!e,headers:{},postBodyFormat:"",postBody:"",compress:P("compress_gel")||P("compress_gel_lr")};Xt()&&(a.headers["X-Goog-Request-Time"]=JSON.stringify(Math.round(R())));return a}
function Qt(a,b,c){Xt()||(a.requestTimeMs=String(b));P("unsplit_gel_payloads_in_logs")&&(a.unsplitGelPayloadsInLogs=!0);!c&&(b=O("EVENT_ID"))&&(c=Yt(),a.serializedClientEventId={serializedEventId:b,clientCounter:String(c)})}
function Wt(a,b,c){Xt()||G(a,2,b);if(!c&&(b=O("EVENT_ID"))){c=Yt();var d=new sm;G(d,1,b);G(d,2,c);H(a,sm,5,d)}}
function Yt(){var a=O("BATCH_CLIENT_COUNTER")||0;a||(a=Math.floor(Math.random()*qt/2));a++;a>qt&&(a=1);Km("BATCH_CLIENT_COUNTER",a);return a}
function Pt(a,b,c){if(c.videoId)var d="VIDEO";else if(c.playlistId)d="PLAYLIST";else return;a.credentialTransferTokenTargetId=c;a.context=a.context||{};a.context.user=a.context.user||{};a.context.user.credentialTransferTokens=[{token:b,scope:d}]}
function Vt(a,b,c){var d=1===fe(c,um)?1:-1;if(Zd(c,d))d=1;else if(c.getPlaylistId())d=2;else return;H(a,tm,4,c);a=ge(a,dl,1)||new dl;c=ge(a,bl,3)||new bl;var e=new al;G(e,2,b);G(e,1,d);le(c,12,al,e);H(a,bl,3,c)}
function Ut(a){for(var b=[],c=0;c<a.length;c++)try{b.push(new pm(a[c]))}catch(d){Pm(new lo("Transport failed to deserialize "+String(a[c])))}return b}
function Dt(a,b){if(B("yt.logging.transport.enableScrapingForTest")){var c=B("yt.logging.transport.scrapedPayloadsForTesting"),d=B("yt.logging.transport.payloadToScrape");b&&(b=B("yt.logging.transport.getScrapedPayloadFromClientEventsFunction").bind(b.payload)())&&c.push(b);if(d&&1<=d.length)for(b=0;b<d.length;b++)if(a&&a.payload[d[b]]){var e=void 0;c.push((null==(e=a)?void 0:e.payload)[d[b]])}z("yt.logging.transport.scrapedPayloadsForTesting",c)}}
function Xt(){return P("use_request_time_ms_header")||P("lr_use_request_time_ms_header")}
function Ht(a,b){return P("transport_use_scheduler")?po(a,b):kn(a,b)}
function Ot(a){P("transport_use_scheduler")?Di.Ba(a):window.clearTimeout(a)}
function St(a){var b,c,d,e,f,g,h,l,m,n;return x(function(q){return 1==q.h?(d=null==(b=a)?void 0:null==(c=b.responseContext)?void 0:c.globalConfigGroup,e=nt(d,Jk),g=null==(f=d)?void 0:f.hotHashData,h=nt(d,Ik),m=null==(l=d)?void 0:l.coldHashData,(n=ht().resolve(new ct(jq)))?g?e?w(q,kq(n,g,e),2):w(q,kq(n,g),2):q.A(2):q.return()):m?h?w(q,lq(n,m,h),0):w(q,lq(n,m),0):q.A(0)})}
;var Zt=y.ytLoggingGelSequenceIdObj_||{};z("ytLoggingGelSequenceIdObj_",Zt);
function $t(a,b,c,d){d=void 0===d?{}:d;var e={},f=Math.round(d.timestamp||R());e.eventTimeMs=f<Number.MAX_SAFE_INTEGER?f:0;e[a]=b;P("enable_unknown_lact_fix_on_html5")&&Ns();a=Qs();e.context={lastActivityMs:String(d.timestamp||!isFinite(a)?-1:a)};P("log_sequence_info_on_gel_web")&&d.sequenceGroup&&(a=e.context,b=d.sequenceGroup,b={index:au(b),groupKey:b},a.sequence=b,d.endOfSequence&&delete Zt[d.sequenceGroup]);(d.sendIsolatedPayload?Kt:Ct)({endpoint:"log_event",payload:e,cttAuthInfo:d.cttAuthInfo,
dangerousLogToVisitorSession:d.dangerousLogToVisitorSession},c)}
function bu(a){It(void 0,void 0,void 0===a?!1:a)}
function au(a){Zt[a]=a in Zt?Zt[a]+1:0;return Zt[a]}
;var cu=[];function Do(a,b,c){c=void 0===c?{}:c;var d=As;O("ytLoggingEventsDefaultDisabled",!1)&&As===As&&(d=null);P("web_all_payloads_via_jspb")?(c.timestamp||(c.timestamp=R()),cu.push({Mc:a,payload:b,options:c})):$t(a,b,d,c)}
;var du=y.ytLoggingGelSequenceIdObj_||{};z("ytLoggingGelSequenceIdObj_",du);
function eu(a,b,c){c=void 0===c?{}:c;var d=Math.round(c.timestamp||R());G(a,1,d<Number.MAX_SAFE_INTEGER?d:0);var e=Qs();d=new om;G(d,1,c.timestamp||!isFinite(e)?-1:e);if(P("log_sequence_info_on_gel_web")&&c.sequenceGroup){e=c.sequenceGroup;var f=au(e),g=new nm;G(g,2,f);G(g,1,e);H(d,nm,3,g);c.endOfSequence&&delete du[c.sequenceGroup]}H(a,om,33,d);(c.sendIsolatedPayload?Mt:Gt)({endpoint:"log_event",payload:a,cttAuthInfo:c.cttAuthInfo,dangerousLogToVisitorSession:c.dangerousLogToVisitorSession},b)}
;function fu(a,b){b=void 0===b?{}:b;var c=!1;O("ytLoggingEventsDefaultDisabled",!1)&&(c=!0);eu(a,c?null:As,b)}
;function gu(a,b,c){var d=new pm;je(d,cm,72,qm,a);c?eu(d,c,b):fu(d,b)}
function hu(a,b,c){var d=new pm;je(d,bm,73,qm,a);c?eu(d,c,b):fu(d,b)}
function iu(a,b,c){var d=new pm;je(d,am,78,qm,a);c?eu(d,c,b):fu(d,b)}
function ju(a,b,c){var d=new pm;je(d,dm,208,qm,a);c?eu(d,c,b):fu(d,b)}
function ku(a,b,c){var d=new pm;je(d,Wl,156,qm,a);c?eu(d,c,b):fu(d,b)}
function lu(a,b,c){var d=new pm;je(d,Yl,215,qm,a);c?eu(d,c,b):fu(d,b)}
;var mu=new Set,nu=0,ou=0,pu=0,qu=[],ru=["PhantomJS","Googlebot","TO STOP THIS SECURITY SCAN go/scan"];function Co(a){su(a)}
function tu(a){su(a,"WARNING")}
function su(a,b,c,d,e,f,g){f=void 0===f?{}:f;f.name=c||O("INNERTUBE_CONTEXT_CLIENT_NAME",1);f.version=d||O("INNERTUBE_CONTEXT_CLIENT_VERSION");var h=f,l=void 0===b?"ERROR":b,m=void 0===g?!1:g;l=void 0===l?"ERROR":l;m=void 0===m?!1:m;if(a){a.hasOwnProperty("level")&&a.level&&(l=a.level);if(P("console_log_js_exceptions")){var n=[];n.push("Name: "+a.name);n.push("Message: "+a.message);a.hasOwnProperty("params")&&n.push("Error Params: "+JSON.stringify(a.params));a.hasOwnProperty("args")&&n.push("Error args: "+
JSON.stringify(a.args));n.push("File name: "+a.fileName);n.push("Stacktrace: "+a.stack);window.console.log(n.join("\n"),a)}if(!(5<=nu)){var q=qu,u=Ne(a),t=u.message||"Unknown Error",A=u.name||"UnknownError",C=u.stack||a.i||"Not available";if(C.startsWith(A+": "+t)){var E=C.split("\n");E.shift();C=E.join("\n")}var M=u.lineNumber||"Not available",Q=u.fileName||"Not available",S=C,V=0;if(a.hasOwnProperty("args")&&a.args&&a.args.length)for(var Z=0;Z<a.args.length&&!(V=Kn(a.args[Z],"params."+Z,h,V),500<=
V);Z++);else if(a.hasOwnProperty("params")&&a.params){var bb=a.params;if("object"===typeof a.params)for(var Qa in bb){if(bb[Qa]){var Da="params."+Qa,sa=Mn(bb[Qa]);h[Da]=sa;V+=Da.length+sa.length;if(500<V)break}}else h.params=Mn(bb)}if(q.length)for(var ia=0;ia<q.length&&!(V=Kn(q[ia],"params.context."+ia,h,V),500<=V);ia++);navigator.vendor&&!h.hasOwnProperty("vendor")&&(h["device.vendor"]=navigator.vendor);var J={message:t,name:A,lineNumber:M,fileName:Q,stack:S,params:h,sampleWeight:1},ff=Number(a.columnNumber);
isNaN(ff)||(J.lineNumber=J.lineNumber+":"+ff);if("IGNORED"===a.level)var Pc=0;else a:{for(var gf=Gn(),wa=r(gf.Ta),kl=wa.next();!kl.done;kl=wa.next()){var sr=kl.value;if(J.message&&J.message.match(sr.Bg)){Pc=sr.weight;break a}}for(var tr=r(gf.Pa),ll=tr.next();!ll.done;ll=tr.next()){var ur=ll.value;if(ur.callback(J)){Pc=ur.weight;break a}}Pc=1}J.sampleWeight=Pc;for(var vr=r(Bn),ml=vr.next();!ml.done;ml=vr.next()){var nl=ml.value;if(nl.lc[J.name])for(var wr=r(nl.lc[J.name]),ol=wr.next();!ol.done;ol=
wr.next()){var xr=ol.value,Dh=J.message.match(xr.regexp);if(Dh){J.params["params.error.original"]=Dh[0];for(var pl=xr.groups,yr={},Nd=0;Nd<pl.length;Nd++)yr[pl[Nd]]=Dh[Nd+1],J.params["params.error."+pl[Nd]]=Dh[Nd+1];J.message=nl.Ic(yr);break}}}J.params||(J.params={});var zr=Gn();J.params["params.errorServiceSignature"]="msg="+zr.Ta.length+"&cb="+zr.Pa.length;J.params["params.serviceWorker"]="false";y.document&&y.document.querySelectorAll&&(J.params["params.fscripts"]=String(document.querySelectorAll("script:not([nonce])").length));
Bb("sample").constructor!==zb&&(J.params["params.fconst"]="true");window.yterr&&"function"===typeof window.yterr&&window.yterr(J);if(0!==J.sampleWeight&&!mu.has(J.message)){if(m&&P("web_enable_error_204"))uu(void 0===l?"ERROR":l,J);else{var Qc=l;Qc=void 0===Qc?"ERROR":Qc;if("ERROR"===Qc){Hn.cb("handleError",J);if(P("record_app_crashed_web")&&0===pu&&1===J.sampleWeight)if(pu++,P("errors_via_jspb")){var ry=new Nl;var Br=G(ry,1,1);if(!P("report_client_error_with_app_crash_ks")){var sy=new Ll,ty=new Kl,
uy=new Jl,vy=new Il;var wy=G(vy,1,J.message);var xy=H(uy,Il,3,wy);var yy=H(ty,Jl,5,xy);var zy=H(sy,Kl,9,yy);H(Br,Ll,4,zy)}var Cr=new pm;je(Cr,Nl,20,qm,Br);fu(Cr)}else{var Dr={appCrashType:"APP_CRASH_TYPE_BREAKPAD"};P("report_client_error_with_app_crash_ks")||(Dr.systemHealth={crashData:{clientError:{logMessage:{message:J.message}}}});Do("appCrashed",Dr)}ou++}else"WARNING"===Qc&&Hn.cb("handleWarning",J);if(P("kevlar_gel_error_routing"))a:{var hf=Qc;if(P("errors_via_jspb")){if(vu())var Fr=void 0;else{var Od=
new Fl;G(Od,1,J.stack);J.fileName&&G(Od,4,J.fileName);var Sb=J.lineNumber&&J.lineNumber.split?J.lineNumber.split(":"):[];0!==Sb.length&&(1!==Sb.length||isNaN(Number(Sb[0]))?2!==Sb.length||isNaN(Number(Sb[0]))||isNaN(Number(Sb[1]))||(me(Od,2,Number(Sb[0])),me(Od,3,Number(Sb[1]))):me(Od,2,Number(Sb[0])));var Rc=new Il;G(Rc,1,J.message);G(Rc,3,J.name);me(Rc,6,J.sampleWeight);"ERROR"===hf?G(Rc,2,2):"WARNING"===hf?G(Rc,2,1):G(Rc,2,0);var ql=new Gl;G(ql,1,!0);je(ql,Fl,3,Hl,Od);var pc=new Dl;G(pc,3,window.location.href);
for(var Gr=O("FEXP_EXPERIMENTS",[]),rl=0;rl<Gr.length;rl++){var Ay=Gr[rl];Sd(pc);de(pc,5,2,!1,!1).push(Ay)}var sl=O("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");if(!Lm()&&sl)for(var Hr=r(Object.keys(sl)),Sc=Hr.next();!Sc.done;Sc=Hr.next()){var Ir=Sc.value,tl=new Cl;G(tl,1,Ir);G(tl,2,String(sl[Ir]));le(pc,4,Cl,tl)}var ul=J.params;if(ul){var Jr=r(Object.keys(ul));for(Sc=Jr.next();!Sc.done;Sc=Jr.next()){var Kr=Sc.value,vl=new Cl;G(vl,1,"client."+Kr);G(vl,2,String(ul[Kr]));le(pc,4,Cl,vl)}}var Lr=O("SERVER_NAME"),
Mr=O("SERVER_VERSION");if(Lr&&Mr){var wl=new Cl;G(wl,1,"server.name");G(wl,2,Lr);le(pc,4,Cl,wl);var xl=new Cl;G(xl,1,"server.version");G(xl,2,Mr);le(pc,4,Cl,xl)}var Eh=new Jl;H(Eh,Dl,1,pc);H(Eh,Gl,2,ql);H(Eh,Il,3,Rc);Fr=Eh}var Nr=Fr;if(!Nr)break a;var Or=new pm;je(Or,Jl,163,qm,Nr);fu(Or)}else{if(vu())var Pr=void 0;else{var jf={stackTrace:J.stack};J.fileName&&(jf.filename=J.fileName);var Tb=J.lineNumber&&J.lineNumber.split?J.lineNumber.split(":"):[];0!==Tb.length&&(1!==Tb.length||isNaN(Number(Tb[0]))?
2!==Tb.length||isNaN(Number(Tb[0]))||isNaN(Number(Tb[1]))||(jf.lineNumber=Number(Tb[0]),jf.columnNumber=Number(Tb[1])):jf.lineNumber=Number(Tb[0]));var yl={level:"ERROR_LEVEL_UNKNOWN",message:J.message,errorClassName:J.name,sampleWeight:J.sampleWeight};"ERROR"===hf?yl.level="ERROR_LEVEL_ERROR":"WARNING"===hf&&(yl.level="ERROR_LEVEL_WARNNING");var By={isObfuscated:!0,browserStackInfo:jf},Pd={pageUrl:window.location.href,kvPairs:[]};O("FEXP_EXPERIMENTS")&&(Pd.experimentIds=O("FEXP_EXPERIMENTS"));var zl=
O("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS");if(!Lm()&&zl)for(var Qr=r(Object.keys(zl)),Tc=Qr.next();!Tc.done;Tc=Qr.next()){var Rr=Tc.value;Pd.kvPairs.push({key:Rr,value:String(zl[Rr])})}var Al=J.params;if(Al){var Sr=r(Object.keys(Al));for(Tc=Sr.next();!Tc.done;Tc=Sr.next()){var Tr=Tc.value;Pd.kvPairs.push({key:"client."+Tr,value:String(Al[Tr])})}}var Ur=O("SERVER_NAME"),Vr=O("SERVER_VERSION");Ur&&Vr&&(Pd.kvPairs.push({key:"server.name",value:Ur}),Pd.kvPairs.push({key:"server.version",value:Vr}));
Pr={errorMetadata:Pd,stackTrace:By,logMessage:yl}}var Wr=Pr;if(!Wr)break a;Do("clientError",Wr)}if("ERROR"===hf||P("errors_flush_gel_always_killswitch"))b:{if(P("web_fp_via_jspb")&&(bu(!0),!P("web_fp_via_jspb_and_json")))break b;bu()}}P("suppress_error_204_logging")||uu(Qc,J)}try{mu.add(J.message)}catch(nA){}nu++}}}}
function vu(){for(var a=r(ru),b=a.next();!b.done;b=a.next())if(An(b.value.toLowerCase()))return!0;return!1}
function uu(a,b){var c=b.params||{};a={urlParams:{a:"logerror",t:"jserror",type:b.name,msg:b.message.substr(0,250),line:b.lineNumber,level:a,"client.name":c.name},postParams:{url:O("PAGE_NAME",window.location.href),file:b.fileName},method:"POST"};c.version&&(a["client.version"]=c.version);if(a.postParams){b.stack&&(a.postParams.stack=b.stack);b=r(Object.keys(c));for(var d=b.next();!d.done;d=b.next())d=d.value,a.postParams["client."+d]=c[d];if(c=O("LATEST_ECATCHER_SERVICE_TRACKING_PARAMS"))for(b=r(Object.keys(c)),
d=b.next();!d.done;d=b.next())d=d.value,a.postParams[d]=c[d];c=O("SERVER_NAME");b=O("SERVER_VERSION");c&&b&&(a.postParams["server.name"]=c,a.postParams["server.version"]=b)}qn(O("ECATCHER_REPORT_HOST","")+"/error_204",a)}
function wu(a){var b=Ja.apply(1,arguments);a.args||(a.args=[]);a.args.push.apply(a.args,ha(b))}
;function xu(){this.register=new Map}
function yu(a){a=r(a.register.values());for(var b=a.next();!b.done;b=a.next())b.value.Fg("ABORTED")}
xu.prototype.clear=function(){yu(this);this.register.clear()};
var zu=new xu;var Au=Date.now().toString();function Bu(){for(var a=Array(16),b=0;16>b;b++){for(var c=Date.now(),d=0;d<c%23;d++)a[b]=Math.random();a[b]=Math.floor(256*Math.random())}if(Au)for(b=1,c=0;c<Au.length;c++)a[b%16]=a[b%16]^a[(b-1)%16]/4^Au.charCodeAt(c),b++;return a}
function Cu(){if(window.crypto&&window.crypto.getRandomValues)try{var a=Array(16),b=new Uint8Array(16);window.crypto.getRandomValues(b);for(var c=0;c<a.length;c++)a[c]=b[c];return a}catch(d){}return Bu()}
function Du(){for(var a=Cu(),b=[],c=0;c<a.length;c++)b.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(a[c]&63));return b.join("")}
;var Eu=y.ytLoggingDocDocumentNonce_;Eu||(Eu=Du(),z("ytLoggingDocDocumentNonce_",Eu));var Fu=Eu;var Gu={Cf:0,Af:1,Bf:2,ag:3,Df:4,gg:5,cg:6,fg:7,dg:8,eg:9,0:"DEFAULT",1:"CHAT",2:"CONVERSATIONS",3:"MINIPLAYER",4:"DIALOG",5:"VOZ",6:"MUSIC_WATCH_TABS",7:"SHARE",8:"PUSH_NOTIFICATIONS",9:"RICH_GRID_WATCH"};function Hu(a){this.h=a}
function Iu(a){return new Hu({trackingParams:a})}
Hu.prototype.getAsJson=function(){var a={};void 0!==this.h.trackingParams?a.trackingParams=this.h.trackingParams:(a.veType=this.h.veType,void 0!==this.h.veCounter&&(a.veCounter=this.h.veCounter),void 0!==this.h.elementIndex&&(a.elementIndex=this.h.elementIndex));void 0!==this.h.dataElement&&(a.dataElement=this.h.dataElement.getAsJson());void 0!==this.h.youtubeData&&(a.youtubeData=this.h.youtubeData);this.h.isCounterfactual&&(a.isCounterfactual=!0);return a};
Hu.prototype.getAsJspb=function(){var a=new Pl;if(void 0!==this.h.trackingParams){var b=this.h.trackingParams;if(null!=b)if("string"===typeof b)b=b?new Cd(b,zd):Ad||(Ad=new Cd(null,zd));else if(b.constructor!==Cd)if(yd(b))b=b.length?new Cd(new Uint8Array(b),zd):Ad||(Ad=new Cd(null,zd));else throw Error();G(a,1,b)}else void 0!==this.h.veType&&me(a,2,this.h.veType),void 0!==this.h.veCounter&&me(a,6,this.h.veCounter),void 0!==this.h.elementIndex&&me(a,3,this.h.elementIndex),this.h.isCounterfactual&&
G(a,5,!0);void 0!==this.h.dataElement&&(b=this.h.dataElement.getAsJspb(),H(a,Pl,7,b));void 0!==this.h.youtubeData&&H(a,Ok,8,this.h.jspbYoutubeData);return a};
Hu.prototype.toString=function(){return JSON.stringify(this.getAsJson())};
Hu.prototype.isClientVe=function(){return!this.h.trackingParams&&!!this.h.veType};function Ju(a){a=void 0===a?0:a;return 0===a?"client-screen-nonce":"client-screen-nonce."+a}
function Ku(a){a=void 0===a?0:a;return 0===a?"ROOT_VE_TYPE":"ROOT_VE_TYPE."+a}
function Lu(a){return O(Ku(void 0===a?0:a))}
z("yt_logging_screen.getRootVeType",Lu);function Mu(a){return(a=Lu(void 0===a?0:a))?new Hu({veType:a,youtubeData:void 0,jspbYoutubeData:void 0}):null}
function Nu(){var a=O("csn-to-ctt-auth-info");a||(a={},Km("csn-to-ctt-auth-info",a));return a}
function Ou(a){a=O(Ju(void 0===a?0:a));if(!a&&!O("USE_CSN_FALLBACK",!0))return null;a||(a="UNDEFINED_CSN");return a?a:null}
z("yt_logging_screen.getCurrentCsn",Ou);function Pu(a){for(var b=r(Object.values(Gu)),c=b.next();!c.done;c=b.next())if(Ou(c.value)===a)return!0;return!1}
function Qu(a,b,c){var d=Nu();(c=Ou(c))&&delete d[c];b&&(d[a]=b)}
function Ru(a){return Nu()[a]}
z("yt_logging_screen.getCttAuthInfo",Ru);function Su(a,b,c,d){c=void 0===c?0:c;if(a!==O(Ju(c))||b!==O(Ku(c)))if(Qu(a,d,c),Km(Ju(c),a),Km(Ku(c),b),b=function(){setTimeout(function(){if(a)if(P("web_time_via_jspb")){var e=new Ql;G(e,1,Fu);G(e,2,a);var f=new pm;je(f,Ql,111,qm,e);fu(f)}else Do("foregroundHeartbeatScreenAssociated",{clientDocumentNonce:Fu,clientScreenNonce:a})},0)},"requestAnimationFrame"in window)try{window.requestAnimationFrame(b)}catch(e){b()}else b()}
z("yt_logging_screen.setCurrentScreen",Su);var Tu=window.yt&&window.yt.msgs_||window.ytcfg&&window.ytcfg.msgs||{};z("yt.msgs_",Tu);function Uu(a){Fm(Tu,arguments)}
;var Vu=[3611,27686,85013,23462,157557,42016,26926,51236,79148,50160,77504,153587,87907,18630,177018,177023,54445,80935,152172,105675,150723,37521,147285,47786,98349,168271,168954,168277,168273,168270,123695,6827,29434,171388,7282,124448,32276,76278,147868,147869,93911,106531,27259,27262,27263,21759,160866,171243,160789,171244,171241,171245,171242,175492,175493,175494,175495,175496,175497,38408,175498,175503,175504,175505,175506,175507,175508,80637,68727,68728,80353,80356,74610,45707,83962,83970,46713,
166591,89711,74612,155792,93265,74611,131380,128979,139311,128978,131391,105350,139312,134800,131392,113533,93252,99357,94521,114252,113532,94522,94583,88E3,139580,93253,93254,94387,94388,93255,97424,72502,110111,76019,117092,117093,89431,110466,77240,60508,148123,148124,137401,137402,137046,73393,113534,92098,131381,84517,83759,162711,162712,80357,86113,72598,168413,72733,107349,124275,118203,133275,160157,152569,156651,133274,160159,160158,133272,133273,133276,144507,143247,175994,156652,143248,
143249,143250,143251,156653,144401,117431,133797,153964,128572,133405,117429,117430,174734,117432,173996,173995,174953,173994,173997,120080,117259,156655,156654,121692,145656,156656,145655,145653,145654,145657,132972,133051,133658,132971,97615,143359,143356,143361,143358,143360,143357,142303,143353,172859,143354,144479,143355,31402,133624,146477,133623,133622,133621,84774,160801,95117,172721,150497,98930,98931,98932,153320,153321,43347,129889,149123,45474,100352,98443,117985,74613,155911,74614,64502,
136032,74615,74616,122224,74617,77820,74618,93278,93274,93275,93276,22110,29433,133798,132295,120541,82047,113550,75836,75837,42352,84512,76065,75989,51879,16623,32594,27240,32633,74858,156999,3945,16989,45520,25488,25492,25494,55760,14057,18451,57204,57203,17897,18198,17898,17909,43980,46220,11721,147994,49954,96369,3854,151633,56251,159108,25624,152036,16906,99999,68172,47973,72773,26970,26971,96805,17752,73233,109512,22256,14115,22696,89278,89277,109513,43278,43459,43464,89279,43717,55764,22255,
147912,89281,40963,43277,43442,91824,120137,96367,36850,72694,37414,36851,124863,121343,73491,54473,166861,43375,46674,143815,139095,144402,149968,149969,32473,72901,72906,50612,50613,50942,84938,84943,84939,84941,84944,84940,84942,35585,51926,79983,18921,57893,41182,135732,33424,22207,36229,22206,22205,44763,33427,67793,22182,37091,34650,50617,22287,25144,97917,62397,150871,150874,125598,137935,36961,108035,27426,27857,27846,27854,69692,61411,39299,38696,62520,36382,108701,50663,36387,14908,37533,
105443,61635,62274,161670,133818,65702,65703,65701,76256,166382,37671,49953,36216,28237,173718,39553,29222,26107,38050,26108,120745,26109,26110,66881,28236,14586,160598,57929,74723,44098,173689,44099,23528,61699,134104,134103,59149,173191,173192,173193,101951,171502,97346,118051,95102,64882,119505,63595,63349,95101,75240,27039,68823,21537,83464,75707,170215,83113,101952,101953,79610,125755,24402,24400,32925,57173,156421,122502,145268,138480,64423,64424,33986,100828,129089,21409,135155,135156,135157,
135158,158225,135159,135160,167651,135161,135162,135163,158226,158227,135164,135165,135166,11070,11074,17880,30709,30707,30711,30710,30708,146143,63648,63649,111059,5754,20445,151308,151152,130975,130976,167637,110386,113746,66557,17310,28631,21589,164817,168011,154946,68012,162617,60480,138664,141121,164502,31571,141978,150105,150106,150107,150108,76980,41577,45469,38669,13768,13777,141842,62985,4724,59369,43927,43928,12924,100355,56219,27669,10337,47896,122629,139723,139722,121258,107598,127991,
96639,107536,130169,96661,145188,96658,116646,159428,168611,168612,121122,96660,127738,127083,155281,162959,163566,147842,104443,96659,147595,106442,162776,134840,63667,63668,63669,130686,147036,78314,147799,174049,148649,55761,127098,134841,96368,67374,48992,146176,176105,49956,31961,26388,23811,5E4,126250,96370,47355,47356,37935,45521,21760,83769,49977,49974,93497,93498,34325,140759,115803,123707,100081,35309,68314,25602,100339,170873,143516,59018,18248,50625,9729,37168,37169,21667,16749,18635,
39305,18046,53969,8213,93926,102852,110099,22678,69076,137575,139224,100856,154430,17736,3832,147111,55759,64031,93044,93045,170701,170702,34388,167841,170419,17657,17655,39579,39578,170412,77448,8196,11357,69877,8197,168501,156512,161613,156509,161612,161614,82039];function Wu(){var a=tb(Xu),b;return(new Pf(function(c,d){a.onSuccess=function(e){jn(e)?c(new Yu(e)):d(new Zu("Request failed, status="+(e&&"status"in e?e.status:-1),"net.badstatus",e))};
a.onError=function(e){d(new Zu("Unknown request error","net.unknown",e))};
a.onTimeout=function(e){d(new Zu("Request timed out","net.timeout",e))};
b=qn("//googleads.g.doubleclick.net/pagead/id",a)})).sc(function(c){c instanceof Wf&&b.abort();
return Uf(c)})}
function Zu(a,b,c){$a.call(this,a+", errorCode="+b);this.errorCode=b;this.xhr=c;this.name="PromiseAjaxError"}
v(Zu,$a);function Yu(a){this.xhr=a}
;function $u(){this.i=0;this.h=null}
$u.prototype.then=function(a,b,c){return 1===this.i&&a?(a=a.call(c,this.h))&&"function"===typeof a.then?a:av(a):2===this.i&&b?(a=b.call(c,this.h))&&"function"===typeof a.then?a:bv(a):this};
$u.prototype.getValue=function(){return this.h};
$u.prototype.$goog_Thenable=!0;function bv(a){var b=new $u;a=void 0===a?null:a;b.i=2;b.h=void 0===a?null:a;return b}
function av(a){var b=new $u;a=void 0===a?null:a;b.i=1;b.h=void 0===a?null:a;return b}
;function cv(a,b){var c=void 0===c?{}:c;a={method:void 0===b?"POST":b,mode:$m(a)?"same-origin":"cors",credentials:$m(a)?"same-origin":"include"};b={};for(var d=r(Object.keys(c)),e=d.next();!e.done;e=d.next())e=e.value,c[e]&&(b[e]=c[e]);0<Object.keys(b).length&&(a.headers=b);return a}
;function dv(){return Pg()||(jd||kd)&&An("applewebkit")&&!An("version")&&(!An("safari")||An("gsa/"))||$c&&An("version/")?!0:O("EOM_VISITOR_DATA")?!1:!0}
;function ev(a){a:{var b=a.raw_embedded_player_response;if(!b&&(a=a.embedded_player_response))try{b=JSON.parse(a)}catch(d){b="EMBEDDED_PLAYER_MODE_UNKNOWN";break a}if(b)b:{for(var c in il)if(il[c]==b.embeddedPlayerMode){b=il[c];break b}b="EMBEDDED_PLAYER_MODE_UNKNOWN"}else b="EMBEDDED_PLAYER_MODE_UNKNOWN"}return"EMBEDDED_PLAYER_MODE_PFL"===b}
;function fv(a){$a.call(this,a.message||a.description||a.name);this.isMissing=a instanceof gv;this.isTimeout=a instanceof Zu&&"net.timeout"==a.errorCode;this.isCanceled=a instanceof Wf}
v(fv,$a);fv.prototype.name="BiscottiError";function gv(){$a.call(this,"Biscotti ID is missing from server")}
v(gv,$a);gv.prototype.name="BiscottiMissingError";var Xu={format:"RAW",method:"GET",timeout:5E3,withCredentials:!0},hv=null;function iv(){if(P("disable_biscotti_fetch_entirely_for_all_web_clients"))return Error("Biscotti id fetching has been disabled entirely.");if(!dv())return Error("User has not consented - not fetching biscotti id.");var a=O("PLAYER_VARS",{});if("1"==rb(a))return Error("Biscotti ID is not available in private embed mode");if(ev(a))return Error("Biscotti id fetching has been disabled for pfl.")}
function Dm(){var a=iv();if(void 0!==a)return Uf(a);hv||(hv=Wu().then(jv).sc(function(b){return kv(2,b)}));
return hv}
function jv(a){a=a.xhr.responseText;if(0!=a.lastIndexOf(")]}'",0))throw new gv;a=JSON.parse(a.substr(4));if(1<(a.type||1))throw new gv;a=a.id;Em(a);hv=av(a);lv(18E5,2);return a}
function kv(a,b){b=new fv(b);Em("");hv=bv(b);0<a&&lv(12E4,a-1);throw b;}
function lv(a,b){kn(function(){Wu().then(jv,function(c){return kv(b,c)}).sc(cb)},a)}
function mv(){try{var a=B("yt.ads.biscotti.getId_");return a?a():Dm()}catch(b){return Uf(b)}}
;function nv(a){if("1"!=rb(O("PLAYER_VARS",{}))){a&&Cm();try{mv().then(function(){},function(){}),kn(nv,18E5)}catch(b){Pm(b)}}}
;function ov(){var a=Zn(),b=bo(119),c=1<window.devicePixelRatio;if(document.body&&Ni(document.body,"exp-invert-logo"))if(c&&!Ni(document.body,"inverted-hdpi")){var d=document.body;if(d.classList)d.classList.add("inverted-hdpi");else if(!Ni(d,"inverted-hdpi")){var e=Li(d);Mi(d,e+(0<e.length?" inverted-hdpi":"inverted-hdpi"))}}else!c&&Ni(document.body,"inverted-hdpi")&&Oi();if(b!=c){b="f"+(Math.floor(119/31)+1);d=co(b)||0;d=c?d|67108864:d&-67108865;0===d?delete Wn[b]:(c=d.toString(16),Wn[b]=c.toString());
c=!0;P("web_secure_pref_cookie_killswitch")&&(c=!1);b=a.h;d=[];for(var f in Wn)Wn.hasOwnProperty(f)&&d.push(f+"="+encodeURIComponent(String(Wn[f])));Tn(b,d.join("&"),63072E3,a.i,c)}}
;var pv=new Map([["dark","USER_INTERFACE_THEME_DARK"],["light","USER_INTERFACE_THEME_LIGHT"]]);function qv(){var a=void 0===a?window.location.href:a;if(P("kevlar_disable_theme_param"))return null;uc(vc(5,a));try{var b=Ym(a).theme;return pv.get(b)||null}catch(c){}return null}
;function rv(){this.h={};if(this.i=Vn()){var a=Ng.get("CONSISTENCY",void 0);a&&sv(this,{encryptedTokenJarContents:a})}}
rv.prototype.handleResponse=function(a,b){if(!b)throw Error("request needs to be passed into ConsistencyService");var c,d;b=(null==(c=b.Ka.context)?void 0:null==(d=c.request)?void 0:d.consistencyTokenJars)||[];var e;if(a=null==(e=a.responseContext)?void 0:e.consistencyTokenJar){e=r(b);for(c=e.next();!c.done;c=e.next())delete this.h[c.value.encryptedTokenJarContents];sv(this,a)}};
function sv(a,b){if(b.encryptedTokenJarContents&&(a.h[b.encryptedTokenJarContents]=b,"string"===typeof b.expirationSeconds)){var c=Number(b.expirationSeconds);setTimeout(function(){delete a.h[b.encryptedTokenJarContents]},1E3*c);
a.i&&Tn("CONSISTENCY",b.encryptedTokenJarContents,c,void 0,!0)}}
;var tv=window.location.hostname.split(".").slice(-2).join(".");function uv(){var a=O("LOCATION_PLAYABILITY_TOKEN");"TVHTML5"===O("INNERTUBE_CLIENT_NAME")&&(this.h=vv(this))&&(a=this.h.get("yt-location-playability-token"));a&&(this.locationPlayabilityToken=a,this.i=void 0)}
var wv;function xv(){wv=B("yt.clientLocationService.instance");wv||(wv=new uv,z("yt.clientLocationService.instance",wv));return wv}
k=uv.prototype;k.setLocationOnInnerTubeContext=function(a){a.client||(a.client={});this.i?(a.client.locationInfo||(a.client.locationInfo={}),a.client.locationInfo.latitudeE7=Math.floor(1E7*this.i.coords.latitude),a.client.locationInfo.longitudeE7=Math.floor(1E7*this.i.coords.longitude),a.client.locationInfo.horizontalAccuracyMeters=Math.round(this.i.coords.accuracy),a.client.locationInfo.forceLocationPlayabilityTokenRefresh=!0):this.locationPlayabilityToken&&(a.client.locationPlayabilityToken=this.locationPlayabilityToken)};
k.handleResponse=function(a){var b;a=null==(b=a.responseContext)?void 0:b.locationPlayabilityToken;void 0!==a&&(this.locationPlayabilityToken=a,this.i=void 0,"TVHTML5"===O("INNERTUBE_CLIENT_NAME")?(this.h=vv(this))&&this.h.set("yt-location-playability-token",a,15552E3):Tn("YT_CL",JSON.stringify({loctok:a}),15552E3,tv,!0))};
function vv(a){return void 0===a.h?new uo("yt-client-location"):a.h}
k.clearLocationPlayabilityToken=function(a){"TVHTML5"===a?(this.h=vv(this))&&this.h.remove("yt-location-playability-token"):Un("YT_CL")};
k.getCurrentPositionFromGeolocation=function(){var a=this;if(!(navigator&&navigator.geolocation&&navigator.geolocation.getCurrentPosition))return Promise.reject(Error("Geolocation unsupported"));var b=!1,c=1E4;"MWEB"===O("INNERTUBE_CLIENT_NAME")&&(b=!0,c=15E3);return new Promise(function(d,e){navigator.geolocation.getCurrentPosition(function(f){a.i=f;d(f)},function(f){e(f)},{enableHighAccuracy:b,
maximumAge:0,timeout:c})})};
k.createUnpluggedLocationInfo=function(a){var b={};a=a.coords;if(null==a?0:a.latitude)b.latitudeE7=Math.floor(1E7*a.latitude);if(null==a?0:a.longitude)b.longitudeE7=Math.floor(1E7*a.longitude);if(null==a?0:a.accuracy)b.locationRadiusMeters=Math.round(a.accuracy);return b};function yv(a,b){var c,d=null==(c=nt(a,hl))?void 0:c.signal;if(d&&b.Qb&&(c=b.Qb[d]))return c();var e;if((c=null==(e=nt(a,fl))?void 0:e.request)&&b.ie&&(e=b.ie[c]))return e();for(var f in a)if(b.hd[f]&&(a=b.hd[f]))return a()}
;function zv(a){return function(){return new a}}
;var Av={},Bv=(Av.WEB_UNPLUGGED="^unplugged/",Av.WEB_UNPLUGGED_ONBOARDING="^unplugged/",Av.WEB_UNPLUGGED_OPS="^unplugged/",Av.WEB_UNPLUGGED_PUBLIC="^unplugged/",Av.WEB_CREATOR="^creator/",Av.WEB_KIDS="^kids/",Av.WEB_EXPERIMENTS="^experiments/",Av.WEB_MUSIC="^music/",Av.WEB_REMIX="^music/",Av.WEB_MUSIC_EMBEDDED_PLAYER="^music/",Av.WEB_MUSIC_EMBEDDED_PLAYER="^main_app/|^sfv/",Av);
function Cv(a){var b=void 0===b?"UNKNOWN_INTERFACE":b;if(1===a.length)return a[0];var c=Bv[b];if(c){var d=new RegExp(c),e=r(a);for(c=e.next();!c.done;c=e.next())if(c=c.value,d.exec(c))return c}var f=[];Object.entries(Bv).forEach(function(g){var h=r(g);g=h.next().value;h=h.next().value;b!==g&&f.push(h)});
d=new RegExp(f.join("|"));a.sort(function(g,h){return g.length-h.length});
e=r(a);for(c=e.next();!c.done;c=e.next())if(c=c.value,!d.exec(c))return c;return a[0]}
;function Dv(){}
Dv.prototype.m=function(a,b,c){b=void 0===b?{}:b;c=void 0===c?Qn:c;var d=a.clickTrackingParams,e=this.l,f=!1;f=void 0===f?!1:f;e=void 0===e?!1:e;var g=O("INNERTUBE_CONTEXT");if(g){g=ub(g);P("web_no_tracking_params_in_shell_killswitch")||delete g.clickTracking;g.client||(g.client={});var h=g.client;"MWEB"===h.clientName&&(h.clientFormFactor=O("IS_TABLET")?"LARGE_FORM_FACTOR":"SMALL_FORM_FACTOR");h.screenWidthPoints=window.innerWidth;h.screenHeightPoints=window.innerHeight;h.screenPixelDensity=Math.round(window.devicePixelRatio||
1);h.screenDensityFloat=window.devicePixelRatio||1;h.utcOffsetMinutes=-Math.floor((new Date).getTimezoneOffset());var l=void 0===l?!1:l;Zn();var m="USER_INTERFACE_THEME_LIGHT";bo(165)?m="USER_INTERFACE_THEME_DARK":bo(174)?m="USER_INTERFACE_THEME_LIGHT":!P("kevlar_legacy_browsers")&&window.matchMedia&&window.matchMedia("(prefers-color-scheme)").matches&&window.matchMedia("(prefers-color-scheme: dark)").matches&&(m="USER_INTERFACE_THEME_DARK");l=l?m:qv()||m;h.userInterfaceTheme=l;if(!f){if(l=jo())h.connectionType=
l;P("web_log_effective_connection_type")&&(l=ko())&&(g.client.effectiveConnectionType=l)}var n;if(P("web_log_memory_total_kbytes")&&(null==(n=y.navigator)?0:n.deviceMemory)){var q;n=null==(q=y.navigator)?void 0:q.deviceMemory;g.client.memoryTotalKbytes=""+1E6*n}q=Ym(y.location.href);!P("web_populate_internal_geo_killswitch")&&q.internalcountrycode&&(h.internalGeo=q.internalcountrycode);"MWEB"===h.clientName||"WEB"===h.clientName?(h.mainAppWebInfo={graftUrl:y.location.href},P("kevlar_woffle")&&Rn.h&&
(q=Rn.h,h.mainAppWebInfo.pwaInstallabilityStatus=!q.h&&q.i?"PWA_INSTALLABILITY_STATUS_CAN_BE_INSTALLED":"PWA_INSTALLABILITY_STATUS_UNKNOWN"),h.mainAppWebInfo.webDisplayMode=Sn(),h.mainAppWebInfo.isWebNativeShareAvailable=navigator&&void 0!==navigator.share):"TVHTML5"===h.clientName&&(!P("web_lr_app_quality_killswitch")&&(q=O("LIVING_ROOM_APP_QUALITY"))&&(h.tvAppInfo=Object.assign(h.tvAppInfo||{},{appQuality:q})),q=O("LIVING_ROOM_CERTIFICATION_SCOPE"))&&(h.tvAppInfo=Object.assign(h.tvAppInfo||{},{certificationScope:q}));
if(!P("web_populate_time_zone_itc_killswitch")){b:{if("undefined"!==typeof Intl)try{var u=(new Intl.DateTimeFormat).resolvedOptions().timeZone;break b}catch(bb){}u=void 0}u&&(h.timeZone=u)}(u=en())?h.experimentsToken=u:delete h.experimentsToken;u=fn();rv.h||(rv.h=new rv);h=rv.h.h;q=[];n=0;for(var t in h)q[n++]=h[t];g.request=Object.assign({},g.request,{internalExperimentFlags:u,consistencyTokenJars:q});!P("web_prequest_context_killswitch")&&(t=O("INNERTUBE_CONTEXT_PREQUEST_CONTEXT"))&&(g.request.externalPrequestContext=
t);u=Zn();t=bo(58);u=u.get("gsml","");g.user=Object.assign({},g.user);t&&(g.user.enableSafetyMode=t);u&&(g.user.lockedSafetyMode=!0);P("warm_op_csn_cleanup")?e&&(f=Ou())&&(g.clientScreenNonce=f):!f&&(f=Ou())&&(g.clientScreenNonce=f);d&&(g.clickTracking={clickTrackingParams:d});if(d=B("yt.mdx.remote.remoteClient_"))g.remoteClient=d;xv().setLocationOnInnerTubeContext(g);try{var A=an(),C=A.bid;delete A.bid;g.adSignalsInfo={params:[],bid:C};var E=r(Object.entries(A));for(var M=E.next();!M.done;M=E.next()){var Q=
r(M.value),S=Q.next().value,V=Q.next().value;A=S;C=V;d=void 0;null==(d=g.adSignalsInfo.params)||d.push({key:A,value:""+C})}}catch(bb){su(bb)}E=g}else su(Error("Error: No InnerTubeContext shell provided in ytconfig.")),E={};E={context:E};if(M=this.h(a)){this.i(E,M,b);var Z;b="/youtubei/v1/"+Cv(this.j());(M=null==(Z=nt(a.commandMetadata,gl))?void 0:Z.apiUrl)&&(b=M);Z=b;(b=O("INNERTUBE_HOST_OVERRIDE"))&&(Z=String(b)+String(xc(Z)));b={};b.key=O("INNERTUBE_API_KEY");P("json_condensed_response")&&(b.prettyPrint=
"false");Z=Zm(Z,b||{},!1);a=Object.assign({},{command:a},void 0);a={input:Z,kb:cv(Z),Ka:E,config:a};a.config.Yb?a.config.Yb.identity=c:a.config.Yb={identity:c};return a}su(new lo("Error: Failed to create Request from Command.",a))};
da.Object.defineProperties(Dv.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!1}}});function Ev(){}
v(Ev,Dv);Ev.prototype.m=function(){return{input:"/getDatasyncIdsEndpoint",kb:cv("/getDatasyncIdsEndpoint","GET"),Ka:{}}};
Ev.prototype.j=function(){return[]};
Ev.prototype.h=function(){};
Ev.prototype.i=function(){};var Fv={},Gv=(Fv.GET_DATASYNC_IDS=zv(Ev),Fv);function Hv(a){var b=Ja.apply(1,arguments);if(!Iv(a)||b.some(function(d){return!Iv(d)}))throw Error("Only objects may be merged.");
b=r(b);for(var c=b.next();!c.done;c=b.next())Jv(a,c.value);return a}
function Jv(a,b){for(var c in b)if(Iv(b[c])){if(c in a&&!Iv(a[c]))throw Error("Cannot merge an object into a non-object.");c in a||(a[c]={});Jv(a[c],b[c])}else if(Kv(b[c])){if(c in a&&!Kv(a[c]))throw Error("Cannot merge an array into a non-array.");c in a||(a[c]=[]);Lv(a[c],b[c])}else a[c]=b[c];return a}
function Lv(a,b){b=r(b);for(var c=b.next();!c.done;c=b.next())c=c.value,Iv(c)?a.push(Jv({},c)):Kv(c)?a.push(Lv([],c)):a.push(c);return a}
function Iv(a){return"object"===typeof a&&!Array.isArray(a)}
function Kv(a){return"object"===typeof a&&Array.isArray(a)}
;function Mv(a){var b;(b=B("ytcsi."+(a||"")+"data_"))||(b={tick:{},info:{}},z("ytcsi."+(a||"")+"data_",b));return b}
function Nv(){var a=Mv();a.info||(a.info={});return a.info}
function Ov(a){a=Mv(a);a.metadata||(a.metadata={});return a.metadata}
function Pv(a){a=Mv(a);a.tick||(a.tick={});return a.tick}
function Qv(a){a=Mv(a);if(a.gel){var b=a.gel;b.gelInfos||(b.gelInfos={});b.gelTicks||(b.gelTicks={})}else a.gel={gelTicks:{},gelInfos:{}};return a.gel}
function Rv(a){a=Qv(a);a.gelInfos||(a.gelInfos={});return a.gelInfos}
function Sv(a){var b=Mv(a).nonce;b||(b=Du(),Mv(a).nonce=b);return b}
;function Tv(){var a=B("ytcsi.debug");a||(a=[],z("ytcsi.debug",a),z("ytcsi.reference",{}));return a}
function Uv(a){a=a||"";var b=B("ytcsi.reference");b||(Tv(),b=B("ytcsi.reference"));if(b[a])return b[a];var c=Tv(),d={timerName:a,info:{},tick:{},span:{},jspbInfo:[]};c.push(d);return b[a]=d}
;var T={},Vv=(T.auto_search="LATENCY_ACTION_AUTO_SEARCH",T.ad_to_ad="LATENCY_ACTION_AD_TO_AD",T.ad_to_video="LATENCY_ACTION_AD_TO_VIDEO",T["analytics.explore"]="LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE",T.app_startup="LATENCY_ACTION_APP_STARTUP",T["artist.analytics"]="LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS",T["artist.events"]="LATENCY_ACTION_CREATOR_ARTIST_CONCERTS",T["artist.presskit"]="LATENCY_ACTION_CREATOR_ARTIST_PROFILE",T["asset.claimed_videos"]="LATENCY_ACTION_CREATOR_CMS_ASSET_CLAIMED_VIDEOS",
T["asset.composition"]="LATENCY_ACTION_CREATOR_CMS_ASSET_COMPOSITION",T["asset.embeds"]="LATENCY_ACTION_CREATOR_CMS_ASSET_EMBEDS",T["asset.issues"]="LATENCY_ACTION_CREATOR_CMS_ASSET_ISSUES",T["asset.licenses"]="LATENCY_ACTION_CREATOR_CMS_ASSET_LICENSES",T["asset.metadata"]="LATENCY_ACTION_CREATOR_CMS_ASSET_METADATA",T["asset.ownership"]="LATENCY_ACTION_CREATOR_CMS_ASSET_OWNERSHIP",T["asset.policy"]="LATENCY_ACTION_CREATOR_CMS_ASSET_POLICY",T["asset.references"]="LATENCY_ACTION_CREATOR_CMS_ASSET_REFERENCES",
T["asset.sound_recordings"]="LATENCY_ACTION_CREATOR_CMS_ASSET_SOUND_RECORDINGS",T["song.analytics"]="LATENCY_ACTION_CREATOR_SONG_ANALYTICS",T.browse="LATENCY_ACTION_BROWSE",T.cast_splash="LATENCY_ACTION_CAST_SPLASH",T.channels="LATENCY_ACTION_CHANNELS",T.creator_channel_dashboard="LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD",T["channel.analytics"]="LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS",T["channel.comments"]="LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS",T["channel.content"]="LATENCY_ACTION_CREATOR_POST_LIST",
T["channel.content.promotions"]="LATENCY_ACTION_CREATOR_PROMOTION_LIST",T["channel.copyright"]="LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT",T["channel.editing"]="LATENCY_ACTION_CREATOR_CHANNEL_EDITING",T["channel.monetization"]="LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION",T["channel.music"]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC",T["channel.music_storefront"]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT",T["channel.playlists"]="LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS",T["channel.translations"]=
"LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS",T["channel.videos"]="LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS",T["channel.live_streaming"]="LATENCY_ACTION_CREATOR_LIVE_STREAMING",T.chips="LATENCY_ACTION_CHIPS",T["dialog.copyright_strikes"]="LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES",T["dialog.video_copyright"]="LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT",T["dialog.uploads"]="LATENCY_ACTION_CREATOR_DIALOG_UPLOADS",T.direct_playback="LATENCY_ACTION_DIRECT_PLAYBACK",T.embed="LATENCY_ACTION_EMBED",
T.entity_key_serialization_perf="LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF",T.entity_key_deserialization_perf="LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF",T.explore="LATENCY_ACTION_EXPLORE",T.home="LATENCY_ACTION_HOME",T.library="LATENCY_ACTION_LIBRARY",T.live="LATENCY_ACTION_LIVE",T.live_pagination="LATENCY_ACTION_LIVE_PAGINATION",T.onboarding="LATENCY_ACTION_ONBOARDING",T.owner="LATENCY_ACTION_CREATOR_CMS_DASHBOARD",T["owner.allowlist"]="LATENCY_ACTION_CREATOR_CMS_ALLOWLIST",T["owner.analytics"]=
"LATENCY_ACTION_CREATOR_CMS_ANALYTICS",T["owner.art_tracks"]="LATENCY_ACTION_CREATOR_CMS_ART_TRACKS",T["owner.assets"]="LATENCY_ACTION_CREATOR_CMS_ASSETS",T["owner.asset_groups"]="LATENCY_ACTION_CREATOR_CMS_ASSET_GROUPS",T["owner.campaigns"]="LATENCY_ACTION_CREATOR_CMS_CAMPAIGNS",T["owner.channels"]="LATENCY_ACTION_CREATOR_CMS_CHANNELS",T["owner.claimed_videos"]="LATENCY_ACTION_CREATOR_CMS_CLAIMED_VIDEOS",T["owner.claims"]="LATENCY_ACTION_CREATOR_CMS_MANUAL_CLAIMING",T["owner.claims.manual"]="LATENCY_ACTION_CREATOR_CMS_MANUAL_CLAIMING",
T["owner.delivery"]="LATENCY_ACTION_CREATOR_CMS_CONTENT_DELIVERY",T["owner.issues"]="LATENCY_ACTION_CREATOR_CMS_ISSUES",T["owner.licenses"]="LATENCY_ACTION_CREATOR_CMS_LICENSES",T["owner.pitch_music"]="LATENCY_ACTION_CREATOR_CMS_PITCH_MUSIC",T["owner.policies"]="LATENCY_ACTION_CREATOR_CMS_POLICIES",T["owner.releases"]="LATENCY_ACTION_CREATOR_CMS_RELEASES",T["owner.reports"]="LATENCY_ACTION_CREATOR_CMS_REPORTS",T["owner.videos"]="LATENCY_ACTION_CREATOR_CMS_VIDEOS",T.parent_profile_settings="LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS",
T.parent_tools_collection="LATENCY_ACTION_PARENT_TOOLS_COLLECTION",T.parent_tools_dashboard="LATENCY_ACTION_PARENT_TOOLS_DASHBOARD",T.player_att="LATENCY_ACTION_PLAYER_ATTESTATION",T["post.comments"]="LATENCY_ACTION_CREATOR_POST_COMMENTS",T["post.edit"]="LATENCY_ACTION_CREATOR_POST_EDIT",T.prebuffer="LATENCY_ACTION_PREBUFFER",T.prefetch="LATENCY_ACTION_PREFETCH",T.profile_settings="LATENCY_ACTION_KIDS_PROFILE_SETTINGS",T.profile_switcher="LATENCY_ACTION_LOGIN",T.reel_watch="LATENCY_ACTION_REEL_WATCH",
T.results="LATENCY_ACTION_RESULTS",T["promotion.edit"]="LATENCY_ACTION_CREATOR_PROMOTION_EDIT",T.search_ui="LATENCY_ACTION_SEARCH_UI",T.search_suggest="LATENCY_ACTION_SUGGEST",T.search_zero_state="LATENCY_ACTION_SEARCH_ZERO_STATE",T.secret_code="LATENCY_ACTION_KIDS_SECRET_CODE",T.seek="LATENCY_ACTION_PLAYER_SEEK",T.settings="LATENCY_ACTION_SETTINGS",T.store="LATENCY_ACTION_STORE",T.tenx="LATENCY_ACTION_TENX",T.video_to_ad="LATENCY_ACTION_VIDEO_TO_AD",T.watch="LATENCY_ACTION_WATCH",T.watch_it_again=
"LATENCY_ACTION_KIDS_WATCH_IT_AGAIN",T["watch,watch7"]="LATENCY_ACTION_WATCH",T["watch,watch7_html5"]="LATENCY_ACTION_WATCH",T["watch,watch7ad"]="LATENCY_ACTION_WATCH",T["watch,watch7ad_html5"]="LATENCY_ACTION_WATCH",T.wn_comments="LATENCY_ACTION_LOAD_COMMENTS",T.ww_rqs="LATENCY_ACTION_WHO_IS_WATCHING",T["video.analytics"]="LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS",T["video.claims"]="LATENCY_ACTION_CREATOR_VIDEO_CLAIMS",T["video.comments"]="LATENCY_ACTION_CREATOR_VIDEO_COMMENTS",T["video.copyright"]=
"LATENCY_ACTION_CREATOR_VIDEO_COPYRIGHT",T["video.edit"]="LATENCY_ACTION_CREATOR_VIDEO_EDIT",T["video.editor"]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR",T["video.editor_async"]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC",T["video.live_settings"]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS",T["video.live_streaming"]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING",T["video.monetization"]="LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION",T["video.policy"]="LATENCY_ACTION_CREATOR_VIDEO_POLICY",T["video.rights_management"]=
"LATENCY_ACTION_CREATOR_VIDEO_RIGHTS_MANAGEMENT",T["video.translations"]="LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS",T.voice_assistant="LATENCY_ACTION_VOICE_ASSISTANT",T.cast_load_by_entity_to_watch="LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH",T.networkless_performance="LATENCY_ACTION_NETWORKLESS_PERFORMANCE",T.gel_compression="LATENCY_ACTION_GEL_COMPRESSION",T),U={},Wv=(U.ad_allowed="adTypesAllowed",U.yt_abt="adBreakType",U.ad_cpn="adClientPlaybackNonce",U.ad_docid="adVideoId",U.yt_ad_an="adNetworks",
U.ad_at="adType",U.aida="appInstallDataAgeMs",U.browse_id="browseId",U.p="httpProtocol",U.t="transportProtocol",U.cs="commandSource",U.cpn="clientPlaybackNonce",U.ccs="creatorInfo.creatorCanaryState",U.ctop="creatorInfo.topEntityType",U.csn="clientScreenNonce",U.docid="videoId",U.GetHome_rid="requestIds",U.GetSearch_rid="requestIds",U.GetPlayer_rid="requestIds",U.GetWatchNext_rid="requestIds",U.GetBrowse_rid="requestIds",U.GetLibrary_rid="requestIds",U.is_continuation="isContinuation",U.is_nav="isNavigation",
U.b_p="kabukiInfo.browseParams",U.is_prefetch="kabukiInfo.isPrefetch",U.is_secondary_nav="kabukiInfo.isSecondaryNav",U.nav_type="kabukiInfo.navigationType",U.prev_browse_id="kabukiInfo.prevBrowseId",U.query_source="kabukiInfo.querySource",U.voz_type="kabukiInfo.vozType",U.yt_lt="loadType",U.mver="creatorInfo.measurementVersion",U.yt_ad="isMonetized",U.nr="webInfo.navigationReason",U.nrsu="navigationRequestedSameUrl",U.pnt="performanceNavigationTiming",U.prt="playbackRequiresTap",U.plt="playerInfo.playbackType",
U.pis="playerInfo.playerInitializedState",U.paused="playerInfo.isPausedOnLoad",U.yt_pt="playerType",U.fmt="playerInfo.itag",U.yt_pl="watchInfo.isPlaylist",U.yt_pre="playerInfo.preloadType",U.yt_ad_pr="prerollAllowed",U.pa="previousAction",U.yt_red="isRedSubscriber",U.rce="mwebInfo.responseContentEncoding",U.rc="resourceInfo.resourceCache",U.scrh="screenHeight",U.scrw="screenWidth",U.st="serverTimeMs",U.ssdm="shellStartupDurationMs",U.br_trs="tvInfo.bedrockTriggerState",U.kebqat="kabukiInfo.earlyBrowseRequestInfo.abandonmentType",
U.kebqa="kabukiInfo.earlyBrowseRequestInfo.adopted",U.label="tvInfo.label",U.is_mdx="tvInfo.isMdx",U.preloaded="tvInfo.isPreloaded",U.aac_type="tvInfo.authAccessCredentialType",U.upg_player_vis="playerInfo.visibilityState",U.query="unpluggedInfo.query",U.upg_chip_ids_string="unpluggedInfo.upgChipIdsString",U.yt_vst="videoStreamType",U.vph="viewportHeight",U.vpw="viewportWidth",U.yt_vis="isVisible",U.rcl="mwebInfo.responseContentLength",U.GetSettings_rid="requestIds",U.GetTrending_rid="requestIds",
U.GetMusicSearchSuggestions_rid="requestIds",U.REQUEST_ID="requestIds",U),Xv="isContinuation isNavigation kabukiInfo.earlyBrowseRequestInfo.adopted kabukiInfo.isPrefetch kabukiInfo.isSecondaryNav isMonetized navigationRequestedSameUrl performanceNavigationTiming playerInfo.isPausedOnLoad prerollAllowed isRedSubscriber tvInfo.isMdx tvInfo.isPreloaded isVisible watchInfo.isPlaylist playbackRequiresTap".split(" "),Yv={},Zv=(Yv.ccs="CANARY_STATE_",Yv.mver="MEASUREMENT_VERSION_",Yv.pis="PLAYER_INITIALIZED_STATE_",
Yv.yt_pt="LATENCY_PLAYER_",Yv.pa="LATENCY_ACTION_",Yv.ctop="TOP_ENTITY_TYPE_",Yv.yt_vst="VIDEO_STREAM_TYPE_",Yv),$v="all_vc ap aq c cbr cbrand cbrver cmodel cos cosver cplatform ctheme cver ei l_an l_mm plid srt yt_fss yt_li vpst vpni2 vpil2 icrc icrt pa GetAccountOverview_rid GetHistory_rid cmt d_vpct d_vpnfi d_vpni nsru pc pfa pfeh pftr pnc prerender psc rc start tcrt tcrc ssr vpr vps yt_abt yt_fn yt_fs yt_pft yt_pre yt_pt yt_pvis ytu_pvis yt_ref yt_sts tds".split(" ");
function aw(a){return Vv[a]||"LATENCY_ACTION_UNKNOWN"}
function bw(a,b,c){c=Qv(c);if(c.gelInfos)c.gelInfos[a]=!0;else{var d={};c.gelInfos=(d[a]=!0,d)}if(a.match("_rid")){var e=a.split("_rid")[0];a="REQUEST_ID"}if(a in Wv){c=Wv[a];0<=eb(Xv,c)&&(b=!!b);a in Zv&&"string"===typeof b&&(b=Zv[a]+b.toUpperCase());a=b;b=c.split(".");for(var f=d={},g=0;g<b.length-1;g++){var h=b[g];f[h]={};f=f[h]}f[b[b.length-1]]="requestIds"===c?[{id:a,endpoint:e}]:a;return Hv({},d)}0<=eb($v,a)||tu(new lo("Unknown label logged with GEL CSI",a))}
;var W={LATENCY_ACTION_SHORTS_VIDEO_INGESTION_TRANSCODING:179,LATENCY_ACTION_KIDS_PROFILE_SWITCHER:90,LATENCY_ACTION_OFFLINE_THUMBNAIL_TRANSFER:100,LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC:46,LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR:37,LATENCY_ACTION_SPINNER_DISPLAYED:14,LATENCY_ACTION_PLAYABILITY_CHECK:10,LATENCY_ACTION_PROCESS:9,LATENCY_ACTION_APP_STARTUP:5,LATENCY_ACTION_GEL_COMPRESSION:215,LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE:204,LATENCY_ACTION_COMMERCE_ACTION_COMMAND_RPC:203,LATENCY_ACTION_COMMERCE_TRANSACTION:184,
LATENCY_ACTION_LOG_PAYMENT_SERVER_ANALYTICS_RPC:173,LATENCY_ACTION_GET_PAYMENT_INSTRUMENTS_PARAMS_RPC:172,LATENCY_ACTION_GET_FIX_INSTRUMENT_PARAMS_RPC:171,LATENCY_ACTION_RESUME_SUBSCRIPTION_RPC:170,LATENCY_ACTION_PAUSE_SUBSCRIPTION_RPC:169,LATENCY_ACTION_GET_OFFLINE_UPSELL_RPC:168,LATENCY_ACTION_GET_OFFERS_RPC:167,LATENCY_ACTION_GET_CANCELLATION_YT_FLOW_RPC:166,LATENCY_ACTION_GET_CANCELLATION_FLOW_RPC:165,LATENCY_ACTION_UPDATE_CROSS_DEVICE_OFFLINE_STATE_RPC:164,LATENCY_ACTION_GET_OFFER_DETAILS_RPC:163,
LATENCY_ACTION_CANCEL_RECURRENCE_TRANSACTION_RPC:162,LATENCY_ACTION_GET_TIP_MODULE_RPC:161,LATENCY_ACTION_HANDLE_TRANSACTION_RPC:160,LATENCY_ACTION_COMPLETE_TRANSACTION_RPC:159,LATENCY_ACTION_GET_CART_RPC:158,LATENCY_ACTION_THUMBNAIL_FETCH:156,LATENCY_ACTION_ABANDONED_DIRECT_PLAYBACK:154,LATENCY_ACTION_SHARE_VIDEO:153,LATENCY_ACTION_AD_TO_VIDEO_INT:152,LATENCY_ACTION_ABANDONED_BROWSE:151,LATENCY_ACTION_PLAYER_ROTATION:150,LATENCY_ACTION_GENERIC_WEB_VIEW:183,LATENCY_ACTION_SHOPPING_IN_APP:124,LATENCY_ACTION_PLAYER_ATTESTATION:121,
LATENCY_ACTION_PLAYER_SEEK:119,LATENCY_ACTION_SUPER_STICKER_BUY_FLOW:114,LATENCY_ACTION_DOWNLOADS_DATA_ACCESS:180,LATENCY_ACTION_BLOCKS_PERFORMANCE:148,LATENCY_ACTION_ASSISTANT_QUERY:138,LATENCY_ACTION_ASSISTANT_SETTINGS:137,LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF:129,LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF:128,LATENCY_ACTION_PROOF_OF_ORIGIN_TOKEN_CREATE:127,LATENCY_ACTION_EMBEDS_SDK_INITIALIZATION:123,LATENCY_ACTION_NETWORKLESS_PERFORMANCE:122,LATENCY_ACTION_DOWNLOADS_EXPANSION:133,LATENCY_ACTION_ENTITY_TRANSFORM:131,
LATENCY_ACTION_DOWNLOADS_COMPATIBILITY_LAYER:96,LATENCY_ACTION_EMBEDS_SET_VIDEO:95,LATENCY_ACTION_SETTINGS:93,LATENCY_ACTION_ABANDONED_STARTUP:81,LATENCY_ACTION_MEDIA_BROWSER_ALARM_PLAY:80,LATENCY_ACTION_MEDIA_BROWSER_SEARCH:79,LATENCY_ACTION_MEDIA_BROWSER_LOAD_TREE:78,LATENCY_ACTION_WHO_IS_WATCHING:77,LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH:76,LATENCY_ACTION_LITE_SWITCH_ACCOUNT:73,LATENCY_ACTION_ELEMENTS_PERFORMANCE:70,LATENCY_ACTION_LOCATION_SIGNAL_COLLECTION:69,LATENCY_ACTION_MODIFY_CHANNEL_NOTIFICATION:65,
LATENCY_ACTION_OFFLINE_STORE_START:61,LATENCY_ACTION_REEL_EDITOR:58,LATENCY_ACTION_CHANNEL_SUBSCRIBE:56,LATENCY_ACTION_CHANNEL_PREVIEW:55,LATENCY_ACTION_PREFETCH:52,LATENCY_ACTION_ABANDONED_WATCH:45,LATENCY_ACTION_LOAD_COMMENT_REPLIES:26,LATENCY_ACTION_LOAD_COMMENTS:25,LATENCY_ACTION_EDIT_COMMENT:24,LATENCY_ACTION_NEW_COMMENT:23,LATENCY_ACTION_OFFLINE_SHARING_RECEIVER_PAIRING:19,LATENCY_ACTION_EMBED:18,LATENCY_ACTION_MDX_LAUNCH:15,LATENCY_ACTION_RESOLVE_URL:13,LATENCY_ACTION_CAST_SPLASH:149,LATENCY_ACTION_MDX_CONNECT_TO_SESSION:190,
LATENCY_ACTION_MDX_STREAM_TRANSFER:178,LATENCY_ACTION_MDX_CAST:120,LATENCY_ACTION_MDX_COMMAND:12,LATENCY_ACTION_MOBILE_LIVE_NAV_MDE:231,LATENCY_ACTION_REEL_SELECT_SEGMENT:136,LATENCY_ACTION_ACCELERATED_EFFECTS:145,LATENCY_ACTION_SHORTS_LOAD_PROJECT:234,LATENCY_ACTION_SHORTS_TRIM_TO_EDITOR_TRANSCODING:229,LATENCY_ACTION_EDIT_AUDIO_GEN:182,LATENCY_ACTION_UPLOAD_AUDIO_MIXER:147,LATENCY_ACTION_SHORTS_CLIENT_SIDE_RENDERING:157,LATENCY_ACTION_SHORTS_SEG_IMP_TRANSCODING:146,LATENCY_ACTION_SHORTS_AUDIO_PICKER_PLAYBACK:130,
LATENCY_ACTION_SHORTS_WAVEFORM_DOWNLOAD:125,LATENCY_ACTION_SHORTS_CAMERA_AUDIO_DOWNLOAD:240,LATENCY_ACTION_SHORTS_VIDEO_INGESTION:155,LATENCY_ACTION_SHORTS_GALLERY:107,LATENCY_ACTION_SHORTS_TRIM:105,LATENCY_ACTION_SHORTS_EDIT:104,LATENCY_ACTION_SHORTS_CAMERA:103,LATENCY_ACTION_NON_CREATION_MODES_GLOBAL_ENTRYPOINT:239,LATENCY_ACTION_CREATION_MODES_MODE_SWITCH:236,LATENCY_ACTION_CREATION_MODES_GLOBAL_ENTRYPOINT:235,LATENCY_ACTION_PRODUCER_IMPORT_LOCAL_MEDIA:233,LATENCY_ACTION_PRODUCER_EXPORT_PROJECT:193,
LATENCY_ACTION_PRODUCER_EDITOR:192,LATENCY_ACTION_PARENT_TOOLS_DASHBOARD:102,LATENCY_ACTION_PARENT_TOOLS_COLLECTION:101,LATENCY_ACTION_MUSIC_OFFLINE_PLAYLIST_DETAIL:238,LATENCY_ACTION_MUSIC_OFFLINE_ALBUM_DETAIL:237,LATENCY_ACTION_MUSIC_LOAD_RECOMMENDED_MEDIA_ITEMS:116,LATENCY_ACTION_MUSIC_LOAD_MEDIA_ITEMS:115,LATENCY_ACTION_MUSIC_ALBUM_DETAIL:72,LATENCY_ACTION_MUSIC_PLAYLIST_DETAIL:71,LATENCY_ACTION_STORE:175,LATENCY_ACTION_CHIPS:68,LATENCY_ACTION_SEARCH_ZERO_STATE:67,LATENCY_ACTION_LIVE_PAGINATION:117,
LATENCY_ACTION_LIVE:20,LATENCY_ACTION_PREBUFFER:40,LATENCY_ACTION_TENX:39,LATENCY_ACTION_KIDS_PROFILE_SETTINGS:94,LATENCY_ACTION_KIDS_WATCH_IT_AGAIN:92,LATENCY_ACTION_KIDS_SECRET_CODE:91,LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS:89,LATENCY_ACTION_KIDS_ONBOARDING:88,LATENCY_ACTION_KIDS_VOICE_SEARCH:82,LATENCY_ACTION_KIDS_CURATED_COLLECTION:62,LATENCY_ACTION_KIDS_LIBRARY:53,LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS:38,LATENCY_ACTION_CREATOR_VIDEO_RIGHTS_MANAGEMENT:219,LATENCY_ACTION_CREATOR_VIDEO_POLICY:217,
LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION:74,LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING:141,LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS:142,LATENCY_ACTION_CREATOR_VIDEO_EDITOR_ASYNC:51,LATENCY_ACTION_CREATOR_VIDEO_EDITOR:50,LATENCY_ACTION_CREATOR_VIDEO_EDIT:36,LATENCY_ACTION_CREATOR_VIDEO_COPYRIGHT:218,LATENCY_ACTION_CREATOR_VIDEO_COMMENTS:34,LATENCY_ACTION_CREATOR_VIDEO_CLAIMS:216,LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS:33,LATENCY_ACTION_CREATOR_SONG_ANALYTICS:176,LATENCY_ACTION_CREATOR_PROMOTION_LIST:186,
LATENCY_ACTION_CREATOR_PROMOTION_EDIT:185,LATENCY_ACTION_CREATOR_POST_LIST:112,LATENCY_ACTION_CREATOR_POST_EDIT:110,LATENCY_ACTION_CREATOR_POST_COMMENTS:111,LATENCY_ACTION_CREATOR_LIVE_STREAMING:108,LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT:174,LATENCY_ACTION_CREATOR_DIALOG_UPLOADS:86,LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES:87,LATENCY_ACTION_CREATOR_CMS_VIDEOS:202,LATENCY_ACTION_CREATOR_CMS_REPORTS:201,LATENCY_ACTION_CREATOR_CMS_RELEASES:226,LATENCY_ACTION_CREATOR_CMS_POLICIES:225,LATENCY_ACTION_CREATOR_CMS_PITCH_MUSIC:224,
LATENCY_ACTION_CREATOR_CMS_MANUAL_CLAIMING:200,LATENCY_ACTION_CREATOR_CMS_LICENSES:223,LATENCY_ACTION_CREATOR_CMS_ISSUES:191,LATENCY_ACTION_CREATOR_CMS_DASHBOARD:199,LATENCY_ACTION_CREATOR_CMS_CONTENT_DELIVERY:198,LATENCY_ACTION_CREATOR_CMS_CLAIMED_VIDEOS:197,LATENCY_ACTION_CREATOR_CMS_CHANNELS:196,LATENCY_ACTION_CREATOR_CMS_CAMPAIGNS:222,LATENCY_ACTION_CREATOR_CMS_ASSET_SOUND_RECORDINGS:214,LATENCY_ACTION_CREATOR_CMS_ASSET_REFERENCES:209,LATENCY_ACTION_CREATOR_CMS_ASSET_POLICY:208,LATENCY_ACTION_CREATOR_CMS_ASSET_OWNERSHIP:207,
LATENCY_ACTION_CREATOR_CMS_ASSET_METADATA:205,LATENCY_ACTION_CREATOR_CMS_ASSET_LICENSES:212,LATENCY_ACTION_CREATOR_CMS_ASSET_ISSUES:206,LATENCY_ACTION_CREATOR_CMS_ASSET_GROUPS:221,LATENCY_ACTION_CREATOR_CMS_ASSET_EMBEDS:210,LATENCY_ACTION_CREATOR_CMS_ASSET_COMPOSITION:213,LATENCY_ACTION_CREATOR_CMS_ASSET_CLAIMED_VIDEOS:211,LATENCY_ACTION_CREATOR_CMS_ASSETS:195,LATENCY_ACTION_CREATOR_CMS_ART_TRACKS:220,LATENCY_ACTION_CREATOR_CMS_ANALYTICS:194,LATENCY_ACTION_CREATOR_CMS_ALLOWLIST:227,LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS:32,
LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS:48,LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS:139,LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT:177,LATENCY_ACTION_CREATOR_CHANNEL_MUSIC:99,LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION:43,LATENCY_ACTION_CREATOR_CHANNEL_EDITING:113,LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD:49,LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT:44,LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS:66,LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS:31,LATENCY_ACTION_CREATOR_ARTIST_PROFILE:85,LATENCY_ACTION_CREATOR_ARTIST_CONCERTS:84,
LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS:83,LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE:140,LATENCY_ACTION_EXPERIMENTAL_WATCH_UI:181,LATENCY_ACTION_FINE_SCRUBBING_THUMBNAILS:228,LATENCY_ACTION_STORYBOARD_THUMBNAILS:118,LATENCY_ACTION_SEARCH_THUMBNAILS:59,LATENCY_ACTION_ON_DEVICE_MODEL_DOWNLOAD:54,LATENCY_ACTION_VOICE_ASSISTANT:47,LATENCY_ACTION_SEARCH_UI:35,LATENCY_ACTION_SUGGEST:30,LATENCY_ACTION_AUTO_SEARCH:126,LATENCY_ACTION_DOWNLOADS:98,LATENCY_ACTION_EXPLORE:75,LATENCY_ACTION_VIDEO_LIST:63,LATENCY_ACTION_HOME_RESUME:60,
LATENCY_ACTION_SUBSCRIPTIONS_LIST:57,LATENCY_ACTION_THUMBNAIL_LOAD:42,LATENCY_ACTION_FIRST_THUMBNAIL_LOAD:29,LATENCY_ACTION_SUBSCRIPTIONS_FEED:109,LATENCY_ACTION_SUBSCRIPTIONS:28,LATENCY_ACTION_TRENDING:27,LATENCY_ACTION_LIBRARY:21,LATENCY_ACTION_VIDEO_THUMBNAIL:8,LATENCY_ACTION_SHOW_MORE:7,LATENCY_ACTION_VIDEO_PREVIEW:6,LATENCY_ACTION_AD_TO_AD:22,LATENCY_ACTION_VIDEO_TO_AD:17,LATENCY_ACTION_AD_TO_VIDEO:16,LATENCY_ACTION_DIRECT_PLAYBACK:132,LATENCY_ACTION_PREBUFFER_VIDEO:144,LATENCY_ACTION_PREFETCH_VIDEO:143,
LATENCY_ACTION_STARTUP:106,LATENCY_ACTION_INLINE_TO_WATCH:232,LATENCY_ACTION_MUSIC_IMMERSIVE_WATCH:230,LATENCY_ACTION_ONBOARDING:135,LATENCY_ACTION_LOGIN:97,LATENCY_ACTION_REEL_WATCH:41,LATENCY_ACTION_WATCH:3,LATENCY_ACTION_RESULTS:2,LATENCY_ACTION_CHANNELS:4,LATENCY_ACTION_HOME:1,LATENCY_ACTION_BROWSE:11,LATENCY_ACTION_USER_ACTION:189,LATENCY_ACTION_INFRASTRUCTURE:188,LATENCY_ACTION_PAGE_NAVIGATION:187,LATENCY_ACTION_UNKNOWN:0};W[W.LATENCY_ACTION_SHORTS_VIDEO_INGESTION_TRANSCODING]="LATENCY_ACTION_SHORTS_VIDEO_INGESTION_TRANSCODING";
W[W.LATENCY_ACTION_KIDS_PROFILE_SWITCHER]="LATENCY_ACTION_KIDS_PROFILE_SWITCHER";W[W.LATENCY_ACTION_OFFLINE_THUMBNAIL_TRANSFER]="LATENCY_ACTION_OFFLINE_THUMBNAIL_TRANSFER";W[W.LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR_ASYNC";W[W.LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR]="LATENCY_ACTION_CREATOR_VIDEO_VIDEO_EDITOR";W[W.LATENCY_ACTION_SPINNER_DISPLAYED]="LATENCY_ACTION_SPINNER_DISPLAYED";W[W.LATENCY_ACTION_PLAYABILITY_CHECK]="LATENCY_ACTION_PLAYABILITY_CHECK";
W[W.LATENCY_ACTION_PROCESS]="LATENCY_ACTION_PROCESS";W[W.LATENCY_ACTION_APP_STARTUP]="LATENCY_ACTION_APP_STARTUP";W[W.LATENCY_ACTION_GEL_COMPRESSION]="LATENCY_ACTION_GEL_COMPRESSION";W[W.LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE]="LATENCY_ACTION_PREMIUM_PAGE_GET_BROWSE";W[W.LATENCY_ACTION_COMMERCE_ACTION_COMMAND_RPC]="LATENCY_ACTION_COMMERCE_ACTION_COMMAND_RPC";W[W.LATENCY_ACTION_COMMERCE_TRANSACTION]="LATENCY_ACTION_COMMERCE_TRANSACTION";W[W.LATENCY_ACTION_LOG_PAYMENT_SERVER_ANALYTICS_RPC]="LATENCY_ACTION_LOG_PAYMENT_SERVER_ANALYTICS_RPC";
W[W.LATENCY_ACTION_GET_PAYMENT_INSTRUMENTS_PARAMS_RPC]="LATENCY_ACTION_GET_PAYMENT_INSTRUMENTS_PARAMS_RPC";W[W.LATENCY_ACTION_GET_FIX_INSTRUMENT_PARAMS_RPC]="LATENCY_ACTION_GET_FIX_INSTRUMENT_PARAMS_RPC";W[W.LATENCY_ACTION_RESUME_SUBSCRIPTION_RPC]="LATENCY_ACTION_RESUME_SUBSCRIPTION_RPC";W[W.LATENCY_ACTION_PAUSE_SUBSCRIPTION_RPC]="LATENCY_ACTION_PAUSE_SUBSCRIPTION_RPC";W[W.LATENCY_ACTION_GET_OFFLINE_UPSELL_RPC]="LATENCY_ACTION_GET_OFFLINE_UPSELL_RPC";W[W.LATENCY_ACTION_GET_OFFERS_RPC]="LATENCY_ACTION_GET_OFFERS_RPC";
W[W.LATENCY_ACTION_GET_CANCELLATION_YT_FLOW_RPC]="LATENCY_ACTION_GET_CANCELLATION_YT_FLOW_RPC";W[W.LATENCY_ACTION_GET_CANCELLATION_FLOW_RPC]="LATENCY_ACTION_GET_CANCELLATION_FLOW_RPC";W[W.LATENCY_ACTION_UPDATE_CROSS_DEVICE_OFFLINE_STATE_RPC]="LATENCY_ACTION_UPDATE_CROSS_DEVICE_OFFLINE_STATE_RPC";W[W.LATENCY_ACTION_GET_OFFER_DETAILS_RPC]="LATENCY_ACTION_GET_OFFER_DETAILS_RPC";W[W.LATENCY_ACTION_CANCEL_RECURRENCE_TRANSACTION_RPC]="LATENCY_ACTION_CANCEL_RECURRENCE_TRANSACTION_RPC";
W[W.LATENCY_ACTION_GET_TIP_MODULE_RPC]="LATENCY_ACTION_GET_TIP_MODULE_RPC";W[W.LATENCY_ACTION_HANDLE_TRANSACTION_RPC]="LATENCY_ACTION_HANDLE_TRANSACTION_RPC";W[W.LATENCY_ACTION_COMPLETE_TRANSACTION_RPC]="LATENCY_ACTION_COMPLETE_TRANSACTION_RPC";W[W.LATENCY_ACTION_GET_CART_RPC]="LATENCY_ACTION_GET_CART_RPC";W[W.LATENCY_ACTION_THUMBNAIL_FETCH]="LATENCY_ACTION_THUMBNAIL_FETCH";W[W.LATENCY_ACTION_ABANDONED_DIRECT_PLAYBACK]="LATENCY_ACTION_ABANDONED_DIRECT_PLAYBACK";W[W.LATENCY_ACTION_SHARE_VIDEO]="LATENCY_ACTION_SHARE_VIDEO";
W[W.LATENCY_ACTION_AD_TO_VIDEO_INT]="LATENCY_ACTION_AD_TO_VIDEO_INT";W[W.LATENCY_ACTION_ABANDONED_BROWSE]="LATENCY_ACTION_ABANDONED_BROWSE";W[W.LATENCY_ACTION_PLAYER_ROTATION]="LATENCY_ACTION_PLAYER_ROTATION";W[W.LATENCY_ACTION_GENERIC_WEB_VIEW]="LATENCY_ACTION_GENERIC_WEB_VIEW";W[W.LATENCY_ACTION_SHOPPING_IN_APP]="LATENCY_ACTION_SHOPPING_IN_APP";W[W.LATENCY_ACTION_PLAYER_ATTESTATION]="LATENCY_ACTION_PLAYER_ATTESTATION";W[W.LATENCY_ACTION_PLAYER_SEEK]="LATENCY_ACTION_PLAYER_SEEK";
W[W.LATENCY_ACTION_SUPER_STICKER_BUY_FLOW]="LATENCY_ACTION_SUPER_STICKER_BUY_FLOW";W[W.LATENCY_ACTION_DOWNLOADS_DATA_ACCESS]="LATENCY_ACTION_DOWNLOADS_DATA_ACCESS";W[W.LATENCY_ACTION_BLOCKS_PERFORMANCE]="LATENCY_ACTION_BLOCKS_PERFORMANCE";W[W.LATENCY_ACTION_ASSISTANT_QUERY]="LATENCY_ACTION_ASSISTANT_QUERY";W[W.LATENCY_ACTION_ASSISTANT_SETTINGS]="LATENCY_ACTION_ASSISTANT_SETTINGS";W[W.LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF]="LATENCY_ACTION_ENTITY_KEY_DESERIALIZATION_PERF";
W[W.LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF]="LATENCY_ACTION_ENTITY_KEY_SERIALIZATION_PERF";W[W.LATENCY_ACTION_PROOF_OF_ORIGIN_TOKEN_CREATE]="LATENCY_ACTION_PROOF_OF_ORIGIN_TOKEN_CREATE";W[W.LATENCY_ACTION_EMBEDS_SDK_INITIALIZATION]="LATENCY_ACTION_EMBEDS_SDK_INITIALIZATION";W[W.LATENCY_ACTION_NETWORKLESS_PERFORMANCE]="LATENCY_ACTION_NETWORKLESS_PERFORMANCE";W[W.LATENCY_ACTION_DOWNLOADS_EXPANSION]="LATENCY_ACTION_DOWNLOADS_EXPANSION";W[W.LATENCY_ACTION_ENTITY_TRANSFORM]="LATENCY_ACTION_ENTITY_TRANSFORM";
W[W.LATENCY_ACTION_DOWNLOADS_COMPATIBILITY_LAYER]="LATENCY_ACTION_DOWNLOADS_COMPATIBILITY_LAYER";W[W.LATENCY_ACTION_EMBEDS_SET_VIDEO]="LATENCY_ACTION_EMBEDS_SET_VIDEO";W[W.LATENCY_ACTION_SETTINGS]="LATENCY_ACTION_SETTINGS";W[W.LATENCY_ACTION_ABANDONED_STARTUP]="LATENCY_ACTION_ABANDONED_STARTUP";W[W.LATENCY_ACTION_MEDIA_BROWSER_ALARM_PLAY]="LATENCY_ACTION_MEDIA_BROWSER_ALARM_PLAY";W[W.LATENCY_ACTION_MEDIA_BROWSER_SEARCH]="LATENCY_ACTION_MEDIA_BROWSER_SEARCH";
W[W.LATENCY_ACTION_MEDIA_BROWSER_LOAD_TREE]="LATENCY_ACTION_MEDIA_BROWSER_LOAD_TREE";W[W.LATENCY_ACTION_WHO_IS_WATCHING]="LATENCY_ACTION_WHO_IS_WATCHING";W[W.LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH]="LATENCY_ACTION_CAST_LOAD_BY_ENTITY_TO_WATCH";W[W.LATENCY_ACTION_LITE_SWITCH_ACCOUNT]="LATENCY_ACTION_LITE_SWITCH_ACCOUNT";W[W.LATENCY_ACTION_ELEMENTS_PERFORMANCE]="LATENCY_ACTION_ELEMENTS_PERFORMANCE";W[W.LATENCY_ACTION_LOCATION_SIGNAL_COLLECTION]="LATENCY_ACTION_LOCATION_SIGNAL_COLLECTION";
W[W.LATENCY_ACTION_MODIFY_CHANNEL_NOTIFICATION]="LATENCY_ACTION_MODIFY_CHANNEL_NOTIFICATION";W[W.LATENCY_ACTION_OFFLINE_STORE_START]="LATENCY_ACTION_OFFLINE_STORE_START";W[W.LATENCY_ACTION_REEL_EDITOR]="LATENCY_ACTION_REEL_EDITOR";W[W.LATENCY_ACTION_CHANNEL_SUBSCRIBE]="LATENCY_ACTION_CHANNEL_SUBSCRIBE";W[W.LATENCY_ACTION_CHANNEL_PREVIEW]="LATENCY_ACTION_CHANNEL_PREVIEW";W[W.LATENCY_ACTION_PREFETCH]="LATENCY_ACTION_PREFETCH";W[W.LATENCY_ACTION_ABANDONED_WATCH]="LATENCY_ACTION_ABANDONED_WATCH";
W[W.LATENCY_ACTION_LOAD_COMMENT_REPLIES]="LATENCY_ACTION_LOAD_COMMENT_REPLIES";W[W.LATENCY_ACTION_LOAD_COMMENTS]="LATENCY_ACTION_LOAD_COMMENTS";W[W.LATENCY_ACTION_EDIT_COMMENT]="LATENCY_ACTION_EDIT_COMMENT";W[W.LATENCY_ACTION_NEW_COMMENT]="LATENCY_ACTION_NEW_COMMENT";W[W.LATENCY_ACTION_OFFLINE_SHARING_RECEIVER_PAIRING]="LATENCY_ACTION_OFFLINE_SHARING_RECEIVER_PAIRING";W[W.LATENCY_ACTION_EMBED]="LATENCY_ACTION_EMBED";W[W.LATENCY_ACTION_MDX_LAUNCH]="LATENCY_ACTION_MDX_LAUNCH";
W[W.LATENCY_ACTION_RESOLVE_URL]="LATENCY_ACTION_RESOLVE_URL";W[W.LATENCY_ACTION_CAST_SPLASH]="LATENCY_ACTION_CAST_SPLASH";W[W.LATENCY_ACTION_MDX_CONNECT_TO_SESSION]="LATENCY_ACTION_MDX_CONNECT_TO_SESSION";W[W.LATENCY_ACTION_MDX_STREAM_TRANSFER]="LATENCY_ACTION_MDX_STREAM_TRANSFER";W[W.LATENCY_ACTION_MDX_CAST]="LATENCY_ACTION_MDX_CAST";W[W.LATENCY_ACTION_MDX_COMMAND]="LATENCY_ACTION_MDX_COMMAND";W[W.LATENCY_ACTION_MOBILE_LIVE_NAV_MDE]="LATENCY_ACTION_MOBILE_LIVE_NAV_MDE";
W[W.LATENCY_ACTION_REEL_SELECT_SEGMENT]="LATENCY_ACTION_REEL_SELECT_SEGMENT";W[W.LATENCY_ACTION_ACCELERATED_EFFECTS]="LATENCY_ACTION_ACCELERATED_EFFECTS";W[W.LATENCY_ACTION_SHORTS_LOAD_PROJECT]="LATENCY_ACTION_SHORTS_LOAD_PROJECT";W[W.LATENCY_ACTION_SHORTS_TRIM_TO_EDITOR_TRANSCODING]="LATENCY_ACTION_SHORTS_TRIM_TO_EDITOR_TRANSCODING";W[W.LATENCY_ACTION_EDIT_AUDIO_GEN]="LATENCY_ACTION_EDIT_AUDIO_GEN";W[W.LATENCY_ACTION_UPLOAD_AUDIO_MIXER]="LATENCY_ACTION_UPLOAD_AUDIO_MIXER";
W[W.LATENCY_ACTION_SHORTS_CLIENT_SIDE_RENDERING]="LATENCY_ACTION_SHORTS_CLIENT_SIDE_RENDERING";W[W.LATENCY_ACTION_SHORTS_SEG_IMP_TRANSCODING]="LATENCY_ACTION_SHORTS_SEG_IMP_TRANSCODING";W[W.LATENCY_ACTION_SHORTS_AUDIO_PICKER_PLAYBACK]="LATENCY_ACTION_SHORTS_AUDIO_PICKER_PLAYBACK";W[W.LATENCY_ACTION_SHORTS_WAVEFORM_DOWNLOAD]="LATENCY_ACTION_SHORTS_WAVEFORM_DOWNLOAD";W[W.LATENCY_ACTION_SHORTS_CAMERA_AUDIO_DOWNLOAD]="LATENCY_ACTION_SHORTS_CAMERA_AUDIO_DOWNLOAD";
W[W.LATENCY_ACTION_SHORTS_VIDEO_INGESTION]="LATENCY_ACTION_SHORTS_VIDEO_INGESTION";W[W.LATENCY_ACTION_SHORTS_GALLERY]="LATENCY_ACTION_SHORTS_GALLERY";W[W.LATENCY_ACTION_SHORTS_TRIM]="LATENCY_ACTION_SHORTS_TRIM";W[W.LATENCY_ACTION_SHORTS_EDIT]="LATENCY_ACTION_SHORTS_EDIT";W[W.LATENCY_ACTION_SHORTS_CAMERA]="LATENCY_ACTION_SHORTS_CAMERA";W[W.LATENCY_ACTION_NON_CREATION_MODES_GLOBAL_ENTRYPOINT]="LATENCY_ACTION_NON_CREATION_MODES_GLOBAL_ENTRYPOINT";W[W.LATENCY_ACTION_CREATION_MODES_MODE_SWITCH]="LATENCY_ACTION_CREATION_MODES_MODE_SWITCH";
W[W.LATENCY_ACTION_CREATION_MODES_GLOBAL_ENTRYPOINT]="LATENCY_ACTION_CREATION_MODES_GLOBAL_ENTRYPOINT";W[W.LATENCY_ACTION_PRODUCER_IMPORT_LOCAL_MEDIA]="LATENCY_ACTION_PRODUCER_IMPORT_LOCAL_MEDIA";W[W.LATENCY_ACTION_PRODUCER_EXPORT_PROJECT]="LATENCY_ACTION_PRODUCER_EXPORT_PROJECT";W[W.LATENCY_ACTION_PRODUCER_EDITOR]="LATENCY_ACTION_PRODUCER_EDITOR";W[W.LATENCY_ACTION_PARENT_TOOLS_DASHBOARD]="LATENCY_ACTION_PARENT_TOOLS_DASHBOARD";W[W.LATENCY_ACTION_PARENT_TOOLS_COLLECTION]="LATENCY_ACTION_PARENT_TOOLS_COLLECTION";
W[W.LATENCY_ACTION_MUSIC_OFFLINE_PLAYLIST_DETAIL]="LATENCY_ACTION_MUSIC_OFFLINE_PLAYLIST_DETAIL";W[W.LATENCY_ACTION_MUSIC_OFFLINE_ALBUM_DETAIL]="LATENCY_ACTION_MUSIC_OFFLINE_ALBUM_DETAIL";W[W.LATENCY_ACTION_MUSIC_LOAD_RECOMMENDED_MEDIA_ITEMS]="LATENCY_ACTION_MUSIC_LOAD_RECOMMENDED_MEDIA_ITEMS";W[W.LATENCY_ACTION_MUSIC_LOAD_MEDIA_ITEMS]="LATENCY_ACTION_MUSIC_LOAD_MEDIA_ITEMS";W[W.LATENCY_ACTION_MUSIC_ALBUM_DETAIL]="LATENCY_ACTION_MUSIC_ALBUM_DETAIL";W[W.LATENCY_ACTION_MUSIC_PLAYLIST_DETAIL]="LATENCY_ACTION_MUSIC_PLAYLIST_DETAIL";
W[W.LATENCY_ACTION_STORE]="LATENCY_ACTION_STORE";W[W.LATENCY_ACTION_CHIPS]="LATENCY_ACTION_CHIPS";W[W.LATENCY_ACTION_SEARCH_ZERO_STATE]="LATENCY_ACTION_SEARCH_ZERO_STATE";W[W.LATENCY_ACTION_LIVE_PAGINATION]="LATENCY_ACTION_LIVE_PAGINATION";W[W.LATENCY_ACTION_LIVE]="LATENCY_ACTION_LIVE";W[W.LATENCY_ACTION_PREBUFFER]="LATENCY_ACTION_PREBUFFER";W[W.LATENCY_ACTION_TENX]="LATENCY_ACTION_TENX";W[W.LATENCY_ACTION_KIDS_PROFILE_SETTINGS]="LATENCY_ACTION_KIDS_PROFILE_SETTINGS";
W[W.LATENCY_ACTION_KIDS_WATCH_IT_AGAIN]="LATENCY_ACTION_KIDS_WATCH_IT_AGAIN";W[W.LATENCY_ACTION_KIDS_SECRET_CODE]="LATENCY_ACTION_KIDS_SECRET_CODE";W[W.LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS]="LATENCY_ACTION_KIDS_PARENT_PROFILE_SETTINGS";W[W.LATENCY_ACTION_KIDS_ONBOARDING]="LATENCY_ACTION_KIDS_ONBOARDING";W[W.LATENCY_ACTION_KIDS_VOICE_SEARCH]="LATENCY_ACTION_KIDS_VOICE_SEARCH";W[W.LATENCY_ACTION_KIDS_CURATED_COLLECTION]="LATENCY_ACTION_KIDS_CURATED_COLLECTION";
W[W.LATENCY_ACTION_KIDS_LIBRARY]="LATENCY_ACTION_KIDS_LIBRARY";W[W.LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS]="LATENCY_ACTION_CREATOR_VIDEO_TRANSLATIONS";W[W.LATENCY_ACTION_CREATOR_VIDEO_RIGHTS_MANAGEMENT]="LATENCY_ACTION_CREATOR_VIDEO_RIGHTS_MANAGEMENT";W[W.LATENCY_ACTION_CREATOR_VIDEO_POLICY]="LATENCY_ACTION_CREATOR_VIDEO_POLICY";W[W.LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION]="LATENCY_ACTION_CREATOR_VIDEO_MONETIZATION";W[W.LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_STREAMING";
W[W.LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS]="LATENCY_ACTION_CREATOR_VIDEO_LIVE_SETTINGS";W[W.LATENCY_ACTION_CREATOR_VIDEO_EDITOR_ASYNC]="LATENCY_ACTION_CREATOR_VIDEO_EDITOR_ASYNC";W[W.LATENCY_ACTION_CREATOR_VIDEO_EDITOR]="LATENCY_ACTION_CREATOR_VIDEO_EDITOR";W[W.LATENCY_ACTION_CREATOR_VIDEO_EDIT]="LATENCY_ACTION_CREATOR_VIDEO_EDIT";W[W.LATENCY_ACTION_CREATOR_VIDEO_COPYRIGHT]="LATENCY_ACTION_CREATOR_VIDEO_COPYRIGHT";W[W.LATENCY_ACTION_CREATOR_VIDEO_COMMENTS]="LATENCY_ACTION_CREATOR_VIDEO_COMMENTS";
W[W.LATENCY_ACTION_CREATOR_VIDEO_CLAIMS]="LATENCY_ACTION_CREATOR_VIDEO_CLAIMS";W[W.LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS]="LATENCY_ACTION_CREATOR_VIDEO_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_SONG_ANALYTICS]="LATENCY_ACTION_CREATOR_SONG_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_PROMOTION_LIST]="LATENCY_ACTION_CREATOR_PROMOTION_LIST";W[W.LATENCY_ACTION_CREATOR_PROMOTION_EDIT]="LATENCY_ACTION_CREATOR_PROMOTION_EDIT";W[W.LATENCY_ACTION_CREATOR_POST_LIST]="LATENCY_ACTION_CREATOR_POST_LIST";
W[W.LATENCY_ACTION_CREATOR_POST_EDIT]="LATENCY_ACTION_CREATOR_POST_EDIT";W[W.LATENCY_ACTION_CREATOR_POST_COMMENTS]="LATENCY_ACTION_CREATOR_POST_COMMENTS";W[W.LATENCY_ACTION_CREATOR_LIVE_STREAMING]="LATENCY_ACTION_CREATOR_LIVE_STREAMING";W[W.LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT]="LATENCY_ACTION_CREATOR_DIALOG_VIDEO_COPYRIGHT";W[W.LATENCY_ACTION_CREATOR_DIALOG_UPLOADS]="LATENCY_ACTION_CREATOR_DIALOG_UPLOADS";W[W.LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES]="LATENCY_ACTION_CREATOR_DIALOG_COPYRIGHT_STRIKES";
W[W.LATENCY_ACTION_CREATOR_CMS_VIDEOS]="LATENCY_ACTION_CREATOR_CMS_VIDEOS";W[W.LATENCY_ACTION_CREATOR_CMS_REPORTS]="LATENCY_ACTION_CREATOR_CMS_REPORTS";W[W.LATENCY_ACTION_CREATOR_CMS_RELEASES]="LATENCY_ACTION_CREATOR_CMS_RELEASES";W[W.LATENCY_ACTION_CREATOR_CMS_POLICIES]="LATENCY_ACTION_CREATOR_CMS_POLICIES";W[W.LATENCY_ACTION_CREATOR_CMS_PITCH_MUSIC]="LATENCY_ACTION_CREATOR_CMS_PITCH_MUSIC";W[W.LATENCY_ACTION_CREATOR_CMS_MANUAL_CLAIMING]="LATENCY_ACTION_CREATOR_CMS_MANUAL_CLAIMING";
W[W.LATENCY_ACTION_CREATOR_CMS_LICENSES]="LATENCY_ACTION_CREATOR_CMS_LICENSES";W[W.LATENCY_ACTION_CREATOR_CMS_ISSUES]="LATENCY_ACTION_CREATOR_CMS_ISSUES";W[W.LATENCY_ACTION_CREATOR_CMS_DASHBOARD]="LATENCY_ACTION_CREATOR_CMS_DASHBOARD";W[W.LATENCY_ACTION_CREATOR_CMS_CONTENT_DELIVERY]="LATENCY_ACTION_CREATOR_CMS_CONTENT_DELIVERY";W[W.LATENCY_ACTION_CREATOR_CMS_CLAIMED_VIDEOS]="LATENCY_ACTION_CREATOR_CMS_CLAIMED_VIDEOS";W[W.LATENCY_ACTION_CREATOR_CMS_CHANNELS]="LATENCY_ACTION_CREATOR_CMS_CHANNELS";
W[W.LATENCY_ACTION_CREATOR_CMS_CAMPAIGNS]="LATENCY_ACTION_CREATOR_CMS_CAMPAIGNS";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_SOUND_RECORDINGS]="LATENCY_ACTION_CREATOR_CMS_ASSET_SOUND_RECORDINGS";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_REFERENCES]="LATENCY_ACTION_CREATOR_CMS_ASSET_REFERENCES";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_POLICY]="LATENCY_ACTION_CREATOR_CMS_ASSET_POLICY";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_OWNERSHIP]="LATENCY_ACTION_CREATOR_CMS_ASSET_OWNERSHIP";
W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_METADATA]="LATENCY_ACTION_CREATOR_CMS_ASSET_METADATA";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_LICENSES]="LATENCY_ACTION_CREATOR_CMS_ASSET_LICENSES";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_ISSUES]="LATENCY_ACTION_CREATOR_CMS_ASSET_ISSUES";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_GROUPS]="LATENCY_ACTION_CREATOR_CMS_ASSET_GROUPS";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_EMBEDS]="LATENCY_ACTION_CREATOR_CMS_ASSET_EMBEDS";W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_COMPOSITION]="LATENCY_ACTION_CREATOR_CMS_ASSET_COMPOSITION";
W[W.LATENCY_ACTION_CREATOR_CMS_ASSET_CLAIMED_VIDEOS]="LATENCY_ACTION_CREATOR_CMS_ASSET_CLAIMED_VIDEOS";W[W.LATENCY_ACTION_CREATOR_CMS_ASSETS]="LATENCY_ACTION_CREATOR_CMS_ASSETS";W[W.LATENCY_ACTION_CREATOR_CMS_ART_TRACKS]="LATENCY_ACTION_CREATOR_CMS_ART_TRACKS";W[W.LATENCY_ACTION_CREATOR_CMS_ANALYTICS]="LATENCY_ACTION_CREATOR_CMS_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_CMS_ALLOWLIST]="LATENCY_ACTION_CREATOR_CMS_ALLOWLIST";W[W.LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS]="LATENCY_ACTION_CREATOR_CHANNEL_VIDEOS";
W[W.LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS]="LATENCY_ACTION_CREATOR_CHANNEL_TRANSLATIONS";W[W.LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS]="LATENCY_ACTION_CREATOR_CHANNEL_PLAYLISTS";W[W.LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC_STOREFRONT";W[W.LATENCY_ACTION_CREATOR_CHANNEL_MUSIC]="LATENCY_ACTION_CREATOR_CHANNEL_MUSIC";W[W.LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION]="LATENCY_ACTION_CREATOR_CHANNEL_MONETIZATION";
W[W.LATENCY_ACTION_CREATOR_CHANNEL_EDITING]="LATENCY_ACTION_CREATOR_CHANNEL_EDITING";W[W.LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD]="LATENCY_ACTION_CREATOR_CHANNEL_DASHBOARD";W[W.LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT]="LATENCY_ACTION_CREATOR_CHANNEL_COPYRIGHT";W[W.LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS]="LATENCY_ACTION_CREATOR_CHANNEL_COMMENTS";W[W.LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS]="LATENCY_ACTION_CREATOR_CHANNEL_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_ARTIST_PROFILE]="LATENCY_ACTION_CREATOR_ARTIST_PROFILE";
W[W.LATENCY_ACTION_CREATOR_ARTIST_CONCERTS]="LATENCY_ACTION_CREATOR_ARTIST_CONCERTS";W[W.LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS]="LATENCY_ACTION_CREATOR_ARTIST_ANALYTICS";W[W.LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE]="LATENCY_ACTION_CREATOR_ANALYTICS_EXPLORE";W[W.LATENCY_ACTION_EXPERIMENTAL_WATCH_UI]="LATENCY_ACTION_EXPERIMENTAL_WATCH_UI";W[W.LATENCY_ACTION_FINE_SCRUBBING_THUMBNAILS]="LATENCY_ACTION_FINE_SCRUBBING_THUMBNAILS";W[W.LATENCY_ACTION_STORYBOARD_THUMBNAILS]="LATENCY_ACTION_STORYBOARD_THUMBNAILS";
W[W.LATENCY_ACTION_SEARCH_THUMBNAILS]="LATENCY_ACTION_SEARCH_THUMBNAILS";W[W.LATENCY_ACTION_ON_DEVICE_MODEL_DOWNLOAD]="LATENCY_ACTION_ON_DEVICE_MODEL_DOWNLOAD";W[W.LATENCY_ACTION_VOICE_ASSISTANT]="LATENCY_ACTION_VOICE_ASSISTANT";W[W.LATENCY_ACTION_SEARCH_UI]="LATENCY_ACTION_SEARCH_UI";W[W.LATENCY_ACTION_SUGGEST]="LATENCY_ACTION_SUGGEST";W[W.LATENCY_ACTION_AUTO_SEARCH]="LATENCY_ACTION_AUTO_SEARCH";W[W.LATENCY_ACTION_DOWNLOADS]="LATENCY_ACTION_DOWNLOADS";W[W.LATENCY_ACTION_EXPLORE]="LATENCY_ACTION_EXPLORE";
W[W.LATENCY_ACTION_VIDEO_LIST]="LATENCY_ACTION_VIDEO_LIST";W[W.LATENCY_ACTION_HOME_RESUME]="LATENCY_ACTION_HOME_RESUME";W[W.LATENCY_ACTION_SUBSCRIPTIONS_LIST]="LATENCY_ACTION_SUBSCRIPTIONS_LIST";W[W.LATENCY_ACTION_THUMBNAIL_LOAD]="LATENCY_ACTION_THUMBNAIL_LOAD";W[W.LATENCY_ACTION_FIRST_THUMBNAIL_LOAD]="LATENCY_ACTION_FIRST_THUMBNAIL_LOAD";W[W.LATENCY_ACTION_SUBSCRIPTIONS_FEED]="LATENCY_ACTION_SUBSCRIPTIONS_FEED";W[W.LATENCY_ACTION_SUBSCRIPTIONS]="LATENCY_ACTION_SUBSCRIPTIONS";
W[W.LATENCY_ACTION_TRENDING]="LATENCY_ACTION_TRENDING";W[W.LATENCY_ACTION_LIBRARY]="LATENCY_ACTION_LIBRARY";W[W.LATENCY_ACTION_VIDEO_THUMBNAIL]="LATENCY_ACTION_VIDEO_THUMBNAIL";W[W.LATENCY_ACTION_SHOW_MORE]="LATENCY_ACTION_SHOW_MORE";W[W.LATENCY_ACTION_VIDEO_PREVIEW]="LATENCY_ACTION_VIDEO_PREVIEW";W[W.LATENCY_ACTION_AD_TO_AD]="LATENCY_ACTION_AD_TO_AD";W[W.LATENCY_ACTION_VIDEO_TO_AD]="LATENCY_ACTION_VIDEO_TO_AD";W[W.LATENCY_ACTION_AD_TO_VIDEO]="LATENCY_ACTION_AD_TO_VIDEO";
W[W.LATENCY_ACTION_DIRECT_PLAYBACK]="LATENCY_ACTION_DIRECT_PLAYBACK";W[W.LATENCY_ACTION_PREBUFFER_VIDEO]="LATENCY_ACTION_PREBUFFER_VIDEO";W[W.LATENCY_ACTION_PREFETCH_VIDEO]="LATENCY_ACTION_PREFETCH_VIDEO";W[W.LATENCY_ACTION_STARTUP]="LATENCY_ACTION_STARTUP";W[W.LATENCY_ACTION_INLINE_TO_WATCH]="LATENCY_ACTION_INLINE_TO_WATCH";W[W.LATENCY_ACTION_MUSIC_IMMERSIVE_WATCH]="LATENCY_ACTION_MUSIC_IMMERSIVE_WATCH";W[W.LATENCY_ACTION_ONBOARDING]="LATENCY_ACTION_ONBOARDING";W[W.LATENCY_ACTION_LOGIN]="LATENCY_ACTION_LOGIN";
W[W.LATENCY_ACTION_REEL_WATCH]="LATENCY_ACTION_REEL_WATCH";W[W.LATENCY_ACTION_WATCH]="LATENCY_ACTION_WATCH";W[W.LATENCY_ACTION_RESULTS]="LATENCY_ACTION_RESULTS";W[W.LATENCY_ACTION_CHANNELS]="LATENCY_ACTION_CHANNELS";W[W.LATENCY_ACTION_HOME]="LATENCY_ACTION_HOME";W[W.LATENCY_ACTION_BROWSE]="LATENCY_ACTION_BROWSE";W[W.LATENCY_ACTION_USER_ACTION]="LATENCY_ACTION_USER_ACTION";W[W.LATENCY_ACTION_INFRASTRUCTURE]="LATENCY_ACTION_INFRASTRUCTURE";W[W.LATENCY_ACTION_PAGE_NAVIGATION]="LATENCY_ACTION_PAGE_NAVIGATION";
W[W.LATENCY_ACTION_UNKNOWN]="LATENCY_ACTION_UNKNOWN";var cw={LATENCY_NETWORK_MOBILE:2,LATENCY_NETWORK_WIFI:1,LATENCY_NETWORK_UNKNOWN:0};cw[cw.LATENCY_NETWORK_MOBILE]="LATENCY_NETWORK_MOBILE";cw[cw.LATENCY_NETWORK_WIFI]="LATENCY_NETWORK_WIFI";cw[cw.LATENCY_NETWORK_UNKNOWN]="LATENCY_NETWORK_UNKNOWN";
var X={CONN_INVALID:31,CONN_CELLULAR_5G_NSA:12,CONN_CELLULAR_5G_SA:11,CONN_WIFI_METERED:10,CONN_CELLULAR_5G:9,CONN_DISCO:8,CONN_CELLULAR_UNKNOWN:7,CONN_CELLULAR_4G:6,CONN_CELLULAR_3G:5,CONN_CELLULAR_2G:4,CONN_WIFI:3,CONN_NONE:2,CONN_UNKNOWN:1,CONN_DEFAULT:0};X[X.CONN_INVALID]="CONN_INVALID";X[X.CONN_CELLULAR_5G_NSA]="CONN_CELLULAR_5G_NSA";X[X.CONN_CELLULAR_5G_SA]="CONN_CELLULAR_5G_SA";X[X.CONN_WIFI_METERED]="CONN_WIFI_METERED";X[X.CONN_CELLULAR_5G]="CONN_CELLULAR_5G";X[X.CONN_DISCO]="CONN_DISCO";
X[X.CONN_CELLULAR_UNKNOWN]="CONN_CELLULAR_UNKNOWN";X[X.CONN_CELLULAR_4G]="CONN_CELLULAR_4G";X[X.CONN_CELLULAR_3G]="CONN_CELLULAR_3G";X[X.CONN_CELLULAR_2G]="CONN_CELLULAR_2G";X[X.CONN_WIFI]="CONN_WIFI";X[X.CONN_NONE]="CONN_NONE";X[X.CONN_UNKNOWN]="CONN_UNKNOWN";X[X.CONN_DEFAULT]="CONN_DEFAULT";
var Y={DETAILED_NETWORK_TYPE_NR_NSA:126,DETAILED_NETWORK_TYPE_NR_SA:125,DETAILED_NETWORK_TYPE_INTERNAL_WIFI_IMPAIRED:124,DETAILED_NETWORK_TYPE_APP_WIFI_HOTSPOT:123,DETAILED_NETWORK_TYPE_DISCONNECTED:122,DETAILED_NETWORK_TYPE_NON_MOBILE_UNKNOWN:121,DETAILED_NETWORK_TYPE_MOBILE_UNKNOWN:120,DETAILED_NETWORK_TYPE_WIMAX:119,DETAILED_NETWORK_TYPE_ETHERNET:118,DETAILED_NETWORK_TYPE_BLUETOOTH:117,DETAILED_NETWORK_TYPE_WIFI:116,DETAILED_NETWORK_TYPE_LTE:115,DETAILED_NETWORK_TYPE_HSPAP:114,DETAILED_NETWORK_TYPE_EHRPD:113,
DETAILED_NETWORK_TYPE_EVDO_B:112,DETAILED_NETWORK_TYPE_UMTS:111,DETAILED_NETWORK_TYPE_IDEN:110,DETAILED_NETWORK_TYPE_HSUPA:109,DETAILED_NETWORK_TYPE_HSPA:108,DETAILED_NETWORK_TYPE_HSDPA:107,DETAILED_NETWORK_TYPE_EVDO_A:106,DETAILED_NETWORK_TYPE_EVDO_0:105,DETAILED_NETWORK_TYPE_CDMA:104,DETAILED_NETWORK_TYPE_1_X_RTT:103,DETAILED_NETWORK_TYPE_GPRS:102,DETAILED_NETWORK_TYPE_EDGE:101,DETAILED_NETWORK_TYPE_UNKNOWN:0};Y[Y.DETAILED_NETWORK_TYPE_NR_NSA]="DETAILED_NETWORK_TYPE_NR_NSA";
Y[Y.DETAILED_NETWORK_TYPE_NR_SA]="DETAILED_NETWORK_TYPE_NR_SA";Y[Y.DETAILED_NETWORK_TYPE_INTERNAL_WIFI_IMPAIRED]="DETAILED_NETWORK_TYPE_INTERNAL_WIFI_IMPAIRED";Y[Y.DETAILED_NETWORK_TYPE_APP_WIFI_HOTSPOT]="DETAILED_NETWORK_TYPE_APP_WIFI_HOTSPOT";Y[Y.DETAILED_NETWORK_TYPE_DISCONNECTED]="DETAILED_NETWORK_TYPE_DISCONNECTED";Y[Y.DETAILED_NETWORK_TYPE_NON_MOBILE_UNKNOWN]="DETAILED_NETWORK_TYPE_NON_MOBILE_UNKNOWN";Y[Y.DETAILED_NETWORK_TYPE_MOBILE_UNKNOWN]="DETAILED_NETWORK_TYPE_MOBILE_UNKNOWN";
Y[Y.DETAILED_NETWORK_TYPE_WIMAX]="DETAILED_NETWORK_TYPE_WIMAX";Y[Y.DETAILED_NETWORK_TYPE_ETHERNET]="DETAILED_NETWORK_TYPE_ETHERNET";Y[Y.DETAILED_NETWORK_TYPE_BLUETOOTH]="DETAILED_NETWORK_TYPE_BLUETOOTH";Y[Y.DETAILED_NETWORK_TYPE_WIFI]="DETAILED_NETWORK_TYPE_WIFI";Y[Y.DETAILED_NETWORK_TYPE_LTE]="DETAILED_NETWORK_TYPE_LTE";Y[Y.DETAILED_NETWORK_TYPE_HSPAP]="DETAILED_NETWORK_TYPE_HSPAP";Y[Y.DETAILED_NETWORK_TYPE_EHRPD]="DETAILED_NETWORK_TYPE_EHRPD";Y[Y.DETAILED_NETWORK_TYPE_EVDO_B]="DETAILED_NETWORK_TYPE_EVDO_B";
Y[Y.DETAILED_NETWORK_TYPE_UMTS]="DETAILED_NETWORK_TYPE_UMTS";Y[Y.DETAILED_NETWORK_TYPE_IDEN]="DETAILED_NETWORK_TYPE_IDEN";Y[Y.DETAILED_NETWORK_TYPE_HSUPA]="DETAILED_NETWORK_TYPE_HSUPA";Y[Y.DETAILED_NETWORK_TYPE_HSPA]="DETAILED_NETWORK_TYPE_HSPA";Y[Y.DETAILED_NETWORK_TYPE_HSDPA]="DETAILED_NETWORK_TYPE_HSDPA";Y[Y.DETAILED_NETWORK_TYPE_EVDO_A]="DETAILED_NETWORK_TYPE_EVDO_A";Y[Y.DETAILED_NETWORK_TYPE_EVDO_0]="DETAILED_NETWORK_TYPE_EVDO_0";Y[Y.DETAILED_NETWORK_TYPE_CDMA]="DETAILED_NETWORK_TYPE_CDMA";
Y[Y.DETAILED_NETWORK_TYPE_1_X_RTT]="DETAILED_NETWORK_TYPE_1_X_RTT";Y[Y.DETAILED_NETWORK_TYPE_GPRS]="DETAILED_NETWORK_TYPE_GPRS";Y[Y.DETAILED_NETWORK_TYPE_EDGE]="DETAILED_NETWORK_TYPE_EDGE";Y[Y.DETAILED_NETWORK_TYPE_UNKNOWN]="DETAILED_NETWORK_TYPE_UNKNOWN";var dw={LATENCY_PLAYER_RTSP:7,LATENCY_PLAYER_HTML5_INLINE:6,LATENCY_PLAYER_HTML5_FULLSCREEN:5,LATENCY_PLAYER_HTML5:4,LATENCY_PLAYER_FRAMEWORK:3,LATENCY_PLAYER_FLASH:2,LATENCY_PLAYER_EXO:1,LATENCY_PLAYER_UNKNOWN:0};dw[dw.LATENCY_PLAYER_RTSP]="LATENCY_PLAYER_RTSP";
dw[dw.LATENCY_PLAYER_HTML5_INLINE]="LATENCY_PLAYER_HTML5_INLINE";dw[dw.LATENCY_PLAYER_HTML5_FULLSCREEN]="LATENCY_PLAYER_HTML5_FULLSCREEN";dw[dw.LATENCY_PLAYER_HTML5]="LATENCY_PLAYER_HTML5";dw[dw.LATENCY_PLAYER_FRAMEWORK]="LATENCY_PLAYER_FRAMEWORK";dw[dw.LATENCY_PLAYER_FLASH]="LATENCY_PLAYER_FLASH";dw[dw.LATENCY_PLAYER_EXO]="LATENCY_PLAYER_EXO";dw[dw.LATENCY_PLAYER_UNKNOWN]="LATENCY_PLAYER_UNKNOWN";
var ew={LATENCY_AD_BREAK_TYPE_POSTROLL:3,LATENCY_AD_BREAK_TYPE_MIDROLL:2,LATENCY_AD_BREAK_TYPE_PREROLL:1,LATENCY_AD_BREAK_TYPE_UNKNOWN:0};ew[ew.LATENCY_AD_BREAK_TYPE_POSTROLL]="LATENCY_AD_BREAK_TYPE_POSTROLL";ew[ew.LATENCY_AD_BREAK_TYPE_MIDROLL]="LATENCY_AD_BREAK_TYPE_MIDROLL";ew[ew.LATENCY_AD_BREAK_TYPE_PREROLL]="LATENCY_AD_BREAK_TYPE_PREROLL";ew[ew.LATENCY_AD_BREAK_TYPE_UNKNOWN]="LATENCY_AD_BREAK_TYPE_UNKNOWN";var fw={LATENCY_ACTION_ERROR_STARTUP_TIMEOUT:1,LATENCY_ACTION_ERROR_UNSPECIFIED:0};
fw[fw.LATENCY_ACTION_ERROR_STARTUP_TIMEOUT]="LATENCY_ACTION_ERROR_STARTUP_TIMEOUT";fw[fw.LATENCY_ACTION_ERROR_UNSPECIFIED]="LATENCY_ACTION_ERROR_UNSPECIFIED";var gw={LIVE_STREAM_MODE_WINDOW:5,LIVE_STREAM_MODE_POST:4,LIVE_STREAM_MODE_LP:3,LIVE_STREAM_MODE_LIVE:2,LIVE_STREAM_MODE_DVR:1,LIVE_STREAM_MODE_UNKNOWN:0};gw[gw.LIVE_STREAM_MODE_WINDOW]="LIVE_STREAM_MODE_WINDOW";gw[gw.LIVE_STREAM_MODE_POST]="LIVE_STREAM_MODE_POST";gw[gw.LIVE_STREAM_MODE_LP]="LIVE_STREAM_MODE_LP";
gw[gw.LIVE_STREAM_MODE_LIVE]="LIVE_STREAM_MODE_LIVE";gw[gw.LIVE_STREAM_MODE_DVR]="LIVE_STREAM_MODE_DVR";gw[gw.LIVE_STREAM_MODE_UNKNOWN]="LIVE_STREAM_MODE_UNKNOWN";var hw={VIDEO_STREAM_TYPE_VOD:3,VIDEO_STREAM_TYPE_DVR:2,VIDEO_STREAM_TYPE_LIVE:1,VIDEO_STREAM_TYPE_UNSPECIFIED:0};hw[hw.VIDEO_STREAM_TYPE_VOD]="VIDEO_STREAM_TYPE_VOD";hw[hw.VIDEO_STREAM_TYPE_DVR]="VIDEO_STREAM_TYPE_DVR";hw[hw.VIDEO_STREAM_TYPE_LIVE]="VIDEO_STREAM_TYPE_LIVE";hw[hw.VIDEO_STREAM_TYPE_UNSPECIFIED]="VIDEO_STREAM_TYPE_UNSPECIFIED";
var iw={YT_IDB_TRANSACTION_TYPE_READ:2,YT_IDB_TRANSACTION_TYPE_WRITE:1,YT_IDB_TRANSACTION_TYPE_UNKNOWN:0};iw[iw.YT_IDB_TRANSACTION_TYPE_READ]="YT_IDB_TRANSACTION_TYPE_READ";iw[iw.YT_IDB_TRANSACTION_TYPE_WRITE]="YT_IDB_TRANSACTION_TYPE_WRITE";iw[iw.YT_IDB_TRANSACTION_TYPE_UNKNOWN]="YT_IDB_TRANSACTION_TYPE_UNKNOWN";var jw={PLAYER_ROTATION_TYPE_PORTRAIT_TO_FULLSCREEN:2,PLAYER_ROTATION_TYPE_FULLSCREEN_TO_PORTRAIT:1,PLAYER_ROTATION_TYPE_UNKNOWN:0};jw[jw.PLAYER_ROTATION_TYPE_PORTRAIT_TO_FULLSCREEN]="PLAYER_ROTATION_TYPE_PORTRAIT_TO_FULLSCREEN";
jw[jw.PLAYER_ROTATION_TYPE_FULLSCREEN_TO_PORTRAIT]="PLAYER_ROTATION_TYPE_FULLSCREEN_TO_PORTRAIT";jw[jw.PLAYER_ROTATION_TYPE_UNKNOWN]="PLAYER_ROTATION_TYPE_UNKNOWN";var kw="actionVisualElement spinnerInfo resourceInfo playerInfo commentInfo mdxInfo watchInfo thumbnailLoadInfo creatorInfo unpluggedInfo reelInfo subscriptionsFeedInfo requestIds mediaBrowserActionInfo musicLoadActionInfo shoppingInfo webViewInfo prefetchInfo accelerationSession commerceInfo inlineToWatchInfo webInfo tvInfo kabukiInfo mwebInfo musicInfo transcodingContext creationModesContext".split(" ");function lw(a,b){xq.call(this,1,arguments);this.timer=b}
v(lw,xq);var mw=new yq("aft-recorded",lw);var nw=y.ytLoggingLatencyUsageStats_||{};z("ytLoggingLatencyUsageStats_",nw);function ow(){this.h=0}
function pw(){ow.h||(ow.h=new ow);return ow.h}
ow.prototype.tick=function(a,b,c,d){qw(this,"tick_"+a+"_"+b)||(c={timestamp:c,cttAuthInfo:d},P("web_csi_via_jspb")?(d=new mm,G(d,1,a),G(d,2,b),a=new pm,je(a,mm,5,qm,d),fu(a,c)):Do("latencyActionTicked",{tickName:a,clientActionNonce:b},c))};
ow.prototype.info=function(a,b,c){var d=Object.keys(a).join("");qw(this,"info_"+d+"_"+b)||(a=Object.assign({},a),a.clientActionNonce=b,Do("latencyActionInfo",a,{cttAuthInfo:c}))};
ow.prototype.jspbInfo=function(a,b,c){for(var d="",e=0;e<a.toJSON().length;e++)void 0!==a.toJSON()[e]&&(d=0===e?d.concat(""+e):d.concat("_"+e));qw(this,"info_"+d+"_"+b)||(G(a,2,b),b={cttAuthInfo:c},c=new pm,je(c,jm,7,qm,a),fu(c,b))};
ow.prototype.span=function(a,b,c){var d=Object.keys(a).join("");qw(this,"span_"+d+"_"+b)||(a.clientActionNonce=b,Do("latencyActionSpan",a,{cttAuthInfo:c}))};
function qw(a,b){nw[b]=nw[b]||{count:0};var c=nw[b];c.count++;c.time=R();a.h||(a.h=po(function(){var d=R(),e;for(e in nw)nw[e]&&6E4<d-nw[e].time&&delete nw[e];a&&(a.h=0)},5E3));
return 5<c.count?(6===c.count&&1>1E5*Math.random()&&(c=new lo("CSI data exceeded logging limit with key",b.split("_")),0<=b.indexOf("plev")||tu(c)),!0):!1}
;var rw=window;function sw(){this.timing={};this.clearResourceTimings=function(){};
this.webkitClearResourceTimings=function(){};
this.mozClearResourceTimings=function(){};
this.msClearResourceTimings=function(){};
this.oClearResourceTimings=function(){}}
function tw(){var a;if(P("csi_use_performance_navigation_timing")||P("csi_use_performance_navigation_timing_tvhtml5")){var b,c,d,e=null==uw?void 0:null==(a=uw.getEntriesByType)?void 0:null==(b=a.call(uw,"navigation"))?void 0:null==(c=b[0])?void 0:null==(d=c.toJSON)?void 0:d.call(c);e?(e.requestStart=vw(e.requestStart),e.responseEnd=vw(e.responseEnd),e.redirectStart=vw(e.redirectStart),e.redirectEnd=vw(e.redirectEnd),e.domainLookupEnd=vw(e.domainLookupEnd),e.connectStart=vw(e.connectStart),e.connectEnd=
vw(e.connectEnd),e.responseStart=vw(e.responseStart),e.secureConnectionStart=vw(e.secureConnectionStart),e.domainLookupStart=vw(e.domainLookupStart),e.isPerformanceNavigationTiming=!0,a=e):a=uw.timing}else a=uw.timing;return a}
function vw(a){return Math.round(ww()+a)}
function ww(){return(P("csi_use_time_origin")||P("csi_use_time_origin_tvhtml5"))&&uw.timeOrigin?Math.floor(uw.timeOrigin):uw.timing.navigationStart}
var uw=rw.performance||rw.mozPerformance||rw.msPerformance||rw.webkitPerformance||new sw;var xw=!1,yw={'script[name="scheduler/scheduler"]':"sj",'script[name="player/base"]':"pj",'link[rel="stylesheet"][name="www-player"]':"pc",'link[rel="stylesheet"][name="player/www-player"]':"pc",'script[name="desktop_polymer/desktop_polymer"]':"dpj",'link[rel="import"][name="desktop_polymer"]':"dph",'script[name="mobile-c3"]':"mcj",'link[rel="stylesheet"][name="mobile-c3"]':"mcc",'script[name="player-plasma-ias-phone/base"]':"mcppj",'script[name="player-plasma-ias-tablet/base"]':"mcptj",'link[rel="stylesheet"][name="mobile-polymer-player-ias"]':"mcpc",
'link[rel="stylesheet"][name="mobile-polymer-player-svg-ias"]':"mcpsc",'script[name="mobile_blazer_core_mod"]':"mbcj",'link[rel="stylesheet"][name="mobile_blazer_css"]':"mbc",'script[name="mobile_blazer_logged_in_users_mod"]':"mbliuj",'script[name="mobile_blazer_logged_out_users_mod"]':"mblouj",'script[name="mobile_blazer_noncore_mod"]':"mbnj","#player_css":"mbpc",'script[name="mobile_blazer_desktopplayer_mod"]':"mbpj",'link[rel="stylesheet"][name="mobile_blazer_tablet_css"]':"mbtc",'script[name="mobile_blazer_watch_mod"]':"mbwj"};
Wa(uw.clearResourceTimings||uw.webkitClearResourceTimings||uw.mozClearResourceTimings||uw.msClearResourceTimings||uw.oClearResourceTimings||cb,uw);function zw(a,b,c,d){if(null!==b){if("yt_lt"===a){var e="string"===typeof b?b:""+b;Ov(c).loadType=e}(a=bw(a,b,c))&&Aw(a,c,d)}}
function Aw(a,b,c){if(!P("web_csi_via_jspb")||(void 0===c?0:c))c=Uv(b||""),Hv(c.info,a),a.loadType&&(c=a.loadType,Ov(b).loadType=c),Hv(Rv(b),a),c=Sv(b),b=Mv(b).cttAuthInfo,pw().info(a,c,b);else{c=new jm;var d=Object.keys(a);a=Object.values(a);for(var e=0;e<d.length;e++){var f=d[e];try{switch(f){case "actionType":G(c,1,W[a[e]]);break;case "clientActionNonce":G(c,2,a[e]);break;case "clientScreenNonce":G(c,4,a[e]);break;case "loadType":G(c,3,a[e]);break;case "isPrewarmedLaunch":G(c,92,a[e]);break;case "isFirstInstall":G(c,
55,a[e]);break;case "networkType":G(c,5,cw[a[e]]);break;case "connectionType":G(c,26,X[a[e]]);break;case "detailedConnectionType":G(c,27,Y[a[e]]);break;case "isVisible":G(c,6,a[e]);break;case "playerType":G(c,7,dw[a[e]]);break;case "clientPlaybackNonce":G(c,8,a[e]);break;case "adClientPlaybackNonce":G(c,28,a[e]);break;case "previousCpn":G(c,77,a[e]);break;case "targetCpn":G(c,76,a[e]);break;case "isMonetized":G(c,9,a[e]);break;case "isPrerollAllowed":G(c,16,a[e]);break;case "isPrerollShown":G(c,17,
a[e]);break;case "adType":G(c,12,a[e]);break;case "adTypesAllowed":G(c,36,a[e]);break;case "adNetworks":G(c,37,a[e]);break;case "previousAction":G(c,13,W[a[e]]);break;case "isRedSubscriber":G(c,14,a[e]);break;case "serverTimeMs":me(c,15,a[e]);break;case "videoId":c.setVideoId(a[e]);break;case "adVideoId":G(c,20,a[e]);break;case "targetVideoId":G(c,78,a[e]);break;case "adBreakType":G(c,21,ew[a[e]]);break;case "isNavigation":G(c,25,a[e]);break;case "viewportHeight":me(c,29,a[e]);break;case "viewportWidth":me(c,
30,a[e]);break;case "screenHeight":me(c,84,a[e]);break;case "screenWidth":me(c,85,a[e]);break;case "browseId":G(c,31,a[e]);break;case "isCacheHit":G(c,32,a[e]);break;case "httpProtocol":G(c,33,a[e]);break;case "transportProtocol":G(c,34,a[e]);break;case "searchQuery":G(c,41,a[e]);break;case "isContinuation":G(c,42,a[e]);break;case "availableProcessors":me(c,43,a[e]);break;case "sdk":G(c,44,a[e]);break;case "isLocalStream":G(c,45,a[e]);break;case "navigationRequestedSameUrl":G(c,64,a[e]);break;case "shellStartupDurationMs":me(c,
70,a[e]);break;case "appInstallDataAgeMs":me(c,73,a[e]);break;case "latencyActionError":G(c,71,fw[a[e]]);break;case "actionStep":me(c,79,a[e]);break;case "jsHeapSizeLimit":G(c,80,a[e]);break;case "totalJsHeapSize":G(c,81,a[e]);break;case "usedJsHeapSize":G(c,82,a[e]);break;case "sourceVideoDurationMs":G(c,90,a[e]);break;case "videoOutputFrames":G(c,93,a[e]);break;case "isResume":G(c,104,a[e]);break;case "debugTicksExcluded":G(c,105,a[e]);break;case "adPrebufferedTimeSecs":me(c,39,a[e]);break;case "isLivestream":G(c,
47,a[e]);break;case "liveStreamMode":G(c,91,gw[a[e]]);break;case "adCpn2":G(c,48,a[e]);break;case "adDaiDriftMillis":G(c,49,a[e]);break;case "videoStreamType":G(c,53,hw[a[e]]);break;case "playbackRequiresTap":G(c,56,a[e]);break;case "performanceNavigationTiming":G(c,67,a[e]);break;case "transactionType":G(c,74,iw[a[e]]);break;case "playerRotationType":G(c,101,jw[a[e]]);break;case "allowedPreroll":G(c,10,a[e]);break;case "shownPreroll":G(c,11,a[e]);break;case "getHomeRequestId":G(c,57,a[e]);break;
case "getSearchRequestId":G(c,60,a[e]);break;case "getPlayerRequestId":G(c,61,a[e]);break;case "getWatchNextRequestId":G(c,62,a[e]);break;case "getBrowseRequestId":G(c,63,a[e]);break;case "getLibraryRequestId":G(c,66,a[e]);break;case "isTransformerEnabledForFeature":G(c,106,a[e]);break;case "sourceVideoFrameCount":G(c,109,a[e]);break;default:kw.includes(f)&&Pm(new lo("Codegen laipb translator asked to translate message field",""+f))}}catch(g){Pm(Error("Codegen laipb translator failed to set "+f))}}Bw(c,
b)}}
function Bw(a,b){var c=qe(a,3);c&&(Ov(b).loadType=c);Uv(b||"").jspbInfo.push(a);c=Sv(b);b=Mv(b).cttAuthInfo;pw().jspbInfo(a,c,b)}
function Cw(a,b,c){if(!b&&"_"!==a[0]){var d=a;uw.mark&&(0==d.lastIndexOf("mark_",0)||(d="mark_"+d),c&&(d+=" ("+c+")"),uw.mark(d))}d=Uv(c||"");d.tick[a]=b||R();if(d.callback&&d.callback[a]){d=r(d.callback[a]);for(var e=d.next();!e.done;e=d.next())e=e.value,e()}d=Qv(c);d.gelTicks&&(d.gelTicks[a]=!0);e=Pv(c);d=b||R();P("log_repeated_ytcsi_ticks")?a in e||(e[a]=d):e[a]=d;e=Sv(c);var f=Mv(c).cttAuthInfo;"_start"===a?(a=pw(),qw(a,"baseline_"+e)||(b={timestamp:b,cttAuthInfo:f},P("web_csi_via_jspb")?(a=new em,
G(a,1,e),e=new pm,je(e,em,6,qm,a),fu(e,b)):Do("latencyActionBaselined",{clientActionNonce:e},b))):pw().tick(a,e,b,f);Dw(c);return d}
function Ew(){var a=document;if("visibilityState"in a)a=a.visibilityState;else{var b=Cs+"VisibilityState";a=b in a?a[b]:void 0}switch(a){case "hidden":return 0;case "visible":return 1;case "prerender":return 2;case "unloaded":return 3;default:return-1}}
function Fw(a){var b=tw(),c=ww(),d=O("CSI_START_TIMESTAMP_MILLIS",0);0<d&&!P("embeds_web_enable_csi_start_override_killswitch")&&(c=d);c&&(Cw("srt",b.responseStart),1!==a.prerender&&Cw("_start",c,void 0));a=Gw();0<a&&Cw("fpt",a);a=tw();a.isPerformanceNavigationTiming&&Aw({performanceNavigationTiming:!0},void 0);Cw("nreqs",a.requestStart,void 0);Cw("nress",a.responseStart,void 0);Cw("nrese",a.responseEnd,void 0);0<a.redirectEnd-a.redirectStart&&(Cw("nrs",a.redirectStart,void 0),Cw("nre",a.redirectEnd,
void 0));0<a.domainLookupEnd-a.domainLookupStart&&(Cw("ndnss",a.domainLookupStart,void 0),Cw("ndnse",a.domainLookupEnd,void 0));0<a.connectEnd-a.connectStart&&(Cw("ntcps",a.connectStart,void 0),Cw("ntcpe",a.connectEnd,void 0));a.secureConnectionStart>=ww()&&0<a.connectEnd-a.secureConnectionStart&&(Cw("nstcps",a.secureConnectionStart,void 0),Cw("ntcpe",a.connectEnd,void 0));uw&&"getEntriesByType"in uw&&Hw()}
function Iw(a,b){a=document.querySelector(a);if(!a)return!1;var c="",d=a.nodeName;"SCRIPT"===d?(c=a.src,c||(c=a.getAttribute("data-timing-href"))&&(c=window.location.protocol+c)):"LINK"===d&&(c=a.href);qc()&&a.setAttribute("nonce",qc());return c?(a=uw.getEntriesByName(c))&&a[0]&&(a=a[0],c=ww(),Cw("rsf_"+b,c+Math.round(a.fetchStart)),Cw("rse_"+b,c+Math.round(a.responseEnd)),void 0!==a.transferSize&&0===a.transferSize)?!0:!1:!1}
function Jw(){var a=[];if(document.querySelector&&uw&&uw.getEntriesByName)for(var b in yw)if(yw.hasOwnProperty(b)){var c=yw[b];Iw(b,c)&&a.push(c)}return a}
function Hw(){var a=window.location.protocol,b=uw.getEntriesByType("resource");b=gb(b,function(c){return 0===c.name.indexOf(a+"//fonts.gstatic.com/s/")});
(b=ib(b,function(c,d){return d.duration>c.duration?d:c},{duration:0}))&&0<b.startTime&&0<b.responseEnd&&(Cw("wffs",vw(b.startTime)),Cw("wffe",vw(b.responseEnd)))}
function Kw(a){var b=Lw("aft",a);if(b)return b;b=O((a||"")+"TIMING_AFT_KEYS",["ol"]);for(var c=b.length,d=0;d<c;d++){var e=Lw(b[d],a);if(e)return e}return NaN}
function Lw(a,b){if(a=Pv(b)[a])return"number"===typeof a?a:a[a.length-1]}
function Dw(a){var b=Lw("_start",a),c=Kw(a);b&&c&&!xw&&(Dq(mw,new lw(Math.round(c-b),a)),xw=!0)}
function Mw(a,b){for(var c=r(Object.keys(b)),d=c.next();!d.done;d=c.next())if(d=d.value,!Object.keys(a).includes(d)||"object"===typeof b[d]&&!Mw(a[d],b[d]))return!1;return!0}
function Gw(){if(uw.getEntriesByType){var a=uw.getEntriesByType("paint");if(a=jb(a,function(b){return"first-paint"===b.name}))return vw(a.startTime)}a=uw.timing;
return a.Ie?Math.max(0,a.Ie):0}
;function Nw(a,b){Om(function(){Uv("").info.actionType=a;b&&Km("TIMING_AFT_KEYS",b);Km("TIMING_ACTION",a);if(P("web_csi_via_jspb")){var c=O("TIMING_INFO",{}),d=new jm;c=r(Object.entries(c));for(var e=c.next();!e.done;e=c.next()){var f=r(e.value);e=f.next().value;f=f.next().value;switch(e){case "GetBrowse_rid":lm(d,hm(gm(e),String(f)));break;case "GetGuide_rid":lm(d,hm(gm(e),String(f)));break;case "GetHome_rid":lm(d,hm(gm(e),String(f)));break;case "GetPlayer_rid":lm(d,hm(gm(e),String(f)));break;case "GetSearch_rid":lm(d,
hm(gm(e),String(f)));break;case "GetSettings_rid":lm(d,hm(gm(e),String(f)));break;case "GetTrending_rid":lm(d,hm(gm(e),String(f)));break;case "GetWatchNext_rid":lm(d,hm(gm(e),String(f)));break;case "yt_red":G(d,14,!!f);break;case "yt_ad":G(d,9,!!f)}}Bw(d);d=new jm;d=G(d,25,!0);d=G(d,1,W[aw(O("TIMING_ACTION"))]);(c=O("PREVIOUS_ACTION"))&&G(d,13,W[aw(c)]);(c=O("CLIENT_PROTOCOL"))&&G(d,33,c);(c=O("CLIENT_TRANSPORT"))&&G(d,34,c);(c=Ou())&&"UNDEFINED_CSN"!==c&&G(d,4,c);c=Ew();1!==c&&-1!==c||G(d,6,!0);
c=Nv();Ov();G(d,3,"cold");Fw(c);c=Jw();if(0<c.length)for(c=r(c),e=c.next();!e.done;e=c.next())e=e.value,f=new im,G(f,1,e),le(d,83,im,f);Bw(d)}else{d=O("TIMING_INFO",{});for(c in d)d.hasOwnProperty(c)&&zw(c,d[c]);d={isNavigation:!0,actionType:aw(O("TIMING_ACTION"))};if(c=O("PREVIOUS_ACTION"))d.previousAction=aw(c);if(c=O("CLIENT_PROTOCOL"))d.httpProtocol=c;if(c=O("CLIENT_TRANSPORT"))d.transportProtocol=c;(c=Ou())&&"UNDEFINED_CSN"!==c&&(d.clientScreenNonce=c);c=Ew();if(1===c||-1===c)d.isVisible=!0;
Ov();Nv();d.loadType="cold";Fw(Nv());c=Jw();if(0<c.length)for(d.resourceInfo=[],c=r(c),e=c.next();!e.done;e=c.next())d.resourceInfo.push({resourceCache:e.value});Aw(d)}d=Nv();c=Rv();if("cold"===Ov().loadType&&("cold"===d.yt_lt||"cold"===c.loadType)){e=Pv();f=Qv();f=f.gelTicks?f.gelTicks:f.gelTicks={};for(var g in e)if(!(g in f))if("number"===typeof e[g])Cw(g,Lw(g));else if(P("log_repeated_ytcsi_ticks"))for(var h=r(e[g]),l=h.next();!l.done;l=h.next())Cw(g.slice(1),l.value);g={};e=!1;f=r(Object.keys(d));
for(h=f.next();!h.done;h=f.next())h=h.value,(h=bw(h,d[h]))&&!Mw(Rv(),h)&&(Hv(c,h),Hv(g,h),e=!0);e&&Aw(g)}z("ytglobal.timingready_",!0);g=O("TIMING_ACTION");B("ytglobal.timingready_")&&g&&Ow()&&Kw()&&Dw()})()}
function Pw(a,b,c,d){Om(zw)(a,b,c,d)}
function Qw(a,b,c){return Om(Cw)(a,b,c)}
function Ow(){return Om(function(){return"_start"in Pv()})()}
function Rw(){Om(function(){var a=Sv();requestAnimationFrame(function(){setTimeout(function(){a===Sv()&&Qw("ol",void 0,void 0)},0)})})()}
var Sw=window;Sw.ytcsi&&(Sw.ytcsi.info=Pw,Sw.ytcsi.tick=Qw);var Tw="tokens consistency mss client_location entities response_received_commands store PLAYER_PRELOAD".split(" "),Uw=["type.googleapis.com/youtube.api.pfiinnertube.YoutubeApiInnertube.BrowseResponse"];function Vw(a,b,c,d){this.m=a;this.U=b;this.l=c;this.j=d;this.i=void 0;this.h=new Map;a.Qb||(a.Qb={});a.Qb=Object.assign({},Gv,a.Qb)}
function Ww(a,b,c,d){if(void 0!==Vw.h){if(d=Vw.h,a=[a!==d.m,b!==d.U,c!==d.l,!1,!1,void 0!==d.i],a.some(function(e){return e}))throw new lo("InnerTubeTransportService is already initialized",a);
}else Vw.h=new Vw(a,b,c,d)}
function Xw(a){var b={signalServiceEndpoint:{signal:"GET_DATASYNC_IDS"}};var c=void 0===c?Qn:c;var d=yv(b,a.m);if(!d)return Uf(new lo("Error: No request builder found for command.",b));var e=d.m(b,void 0,c);return e?new Pf(function(f){var g,h,l;return x(function(m){if(1==m.h){h="cors"===(null==(g=e.kb)?void 0:g.mode)?"cors":void 0;if(a.l.gf){var n=e.config,q;n=null==n?void 0:null==(q=n.Yb)?void 0:q.sessionIndex;q=Pn(0,{sessionIndex:n});l=Object.assign({},Yw(h),q);return m.A(2)}return w(m,Zw(e.config,
h),3)}2!=m.h&&(l=m.i);f($w(a,e,l));m.h=0})}):Uf(new lo("Error: Failed to build request for command.",b))}
function ax(a,b,c){var d;if(b&&!(null==b?0:null==(d=b.Ig)?0:d.Mg)&&a.j){d=r(Tw);for(var e=d.next();!e.done;e=d.next())e=e.value,a.j[e]&&a.j[e].handleResponse(b,c)}}
function $w(a,b,c){var d,e,f,g,h,l,m,n,q,u,t,A,C,E,M,Q,S,V,Z,bb,Qa,Da,sa,ia,J,ff,Pc,gf;return x(function(wa){switch(wa.h){case 1:wa.A(2);break;case 3:if((d=wa.i)&&!d.isExpired())return wa.return(Promise.resolve(d.h()));case 2:if(null==(e=b)?0:null==(f=e.Ka)?0:f.context)for(g=r([]),h=g.next();!h.done;h=g.next())l=h.value,l.Eg(b.Ka.context);if(null==(m=a.i)||!m.Jg(b.input,b.Ka)){wa.A(4);break}return w(wa,a.i.zg(b.input,b.Ka),5);case 5:return n=wa.i,P("kevlar_process_local_innertube_responses_killswitch")||
ax(a,n,b),wa.return(n);case 4:return(t=null==(u=b.config)?void 0:u.Ea)&&a.h.has(t)&&P("web_memoize_inflight_requests")?q=a.h.get(t):(A=JSON.stringify(b.Ka),M=null!=(E=null==(C=b.kb)?void 0:C.headers)?E:{},b.kb=Object.assign({},b.kb,{headers:Object.assign({},M,c)}),Q=Object.assign({},b.kb),"POST"===b.kb.method&&(Q=Object.assign({},Q,{body:A})),(null==(S=b.config)?0:S.Qe)&&Qw(b.config.Qe),V=function(){return a.U.fetch(b.input,Q,b.config)},q=V(),t&&a.h.set(t,q)),w(wa,q,6);
case 6:if((Z=wa.i)&&"error"in Z&&(null==(bb=Z)?0:null==(Qa=bb.error)?0:Qa.details))for(Da=Z.error.details,sa=r(Da),ia=sa.next();!ia.done;ia=sa.next())J=ia.value,(ff=J["@type"])&&-1<Uw.indexOf(ff)&&(delete J["@type"],Z=J);t&&a.h.has(t)&&a.h.delete(t);(null==(Pc=b.config)?0:Pc.Re)&&Qw(b.config.Re);if(Z||null==(gf=a.i)||!gf.rg(b.input,b.Ka)){wa.A(7);break}return w(wa,a.i.yg(b.input,b.Ka),8);case 8:Z=wa.i;case 7:return ax(a,Z,b),wa.return(Z||void 0)}})}
function Zw(a,b){var c,d,e,f;return x(function(g){if(1==g.h){e=null==(c=a)?void 0:null==(d=c.Yb)?void 0:d.sessionIndex;var h=Pn(0,{sessionIndex:e});if(!(h instanceof Pf)){var l=new Pf(cb);Qf(l,2,h);h=l}return w(g,h,2)}f=g.i;return g.return(Promise.resolve(Object.assign({},Yw(b),f)))})}
function Yw(a){var b={"Content-Type":"application/json"};O("EOM_VISITOR_DATA")?b["X-Goog-EOM-Visitor-Id"]=O("EOM_VISITOR_DATA"):O("VISITOR_DATA")&&(b["X-Goog-Visitor-Id"]=O("VISITOR_DATA"));b["X-Youtube-Bootstrap-Logged-In"]=O("LOGGED_IN",!1);"cors"!==a&&((a=O("INNERTUBE_CONTEXT_CLIENT_NAME"))&&(b["X-Youtube-Client-Name"]=a),(a=O("INNERTUBE_CONTEXT_CLIENT_VERSION"))&&(b["X-Youtube-Client-Version"]=a),(a=O("CHROME_CONNECTED_HEADER"))&&(b["X-Youtube-Chrome-Connected"]=a),(a=O("DOMAIN_ADMIN_STATE"))&&
(b["X-Youtube-Domain-Admin-State"]=a));return b}
;var bx=new bt("INNERTUBE_TRANSPORT_TOKEN");var cx=["share/get_web_player_share_panel"],dx=["feedback"],ex=["notification/modify_channel_preference"],fx=["browse/edit_playlist"],gx=["subscription/subscribe"],hx=["subscription/unsubscribe"];function ix(){}
v(ix,Dv);ix.prototype.j=function(){return gx};
ix.prototype.h=function(a){return nt(a,Bm)||void 0};
ix.prototype.i=function(a,b,c){c=void 0===c?{}:c;b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params);c.botguardResponse&&(a.botguardResponse=c.botguardResponse);c.feature&&(a.clientFeature=c.feature)};
da.Object.defineProperties(ix.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function jx(){}
v(jx,Dv);jx.prototype.j=function(){return hx};
jx.prototype.h=function(a){return nt(a,Am)||void 0};
jx.prototype.i=function(a,b){b.channelIds&&(a.channelIds=b.channelIds);b.siloName&&(a.siloName=b.siloName);b.params&&(a.params=b.params)};
da.Object.defineProperties(jx.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function kx(){}
v(kx,Dv);kx.prototype.j=function(){return dx};
kx.prototype.h=function(a){return nt(a,jl)||void 0};
kx.prototype.i=function(a,b,c){a.feedbackTokens=[];b.feedbackToken&&a.feedbackTokens.push(b.feedbackToken);if(b=b.cpn||c.cpn)a.feedbackContext={cpn:b};a.isFeedbackTokenUnencrypted=!!c.is_feedback_token_unencrypted;a.shouldMerge=!1;c.extra_feedback_tokens&&(a.shouldMerge=!0,a.feedbackTokens=a.feedbackTokens.concat(c.extra_feedback_tokens))};
da.Object.defineProperties(kx.prototype,{l:{configurable:!0,enumerable:!0,get:function(){return!0}}});function lx(){}
v(lx,Dv);lx.prototype.j=function(){return ex};
lx.prototype.h=function(a){return nt(a,zm)||void 0};
lx.prototype.i=function(a,b){b.params&&(a.params=b.params);b.secondaryParams&&(a.secondaryParams=b.secondaryParams)};function mx(){}
v(mx,Dv);mx.prototype.j=function(){return fx};
mx.prototype.h=function(a){return nt(a,ym)||void 0};
mx.prototype.i=function(a,b){b.actions&&(a.actions=b.actions);b.params&&(a.params=b.params);b.playlistId&&(a.playlistId=b.playlistId)};function nx(){}
v(nx,Dv);nx.prototype.j=function(){return cx};
nx.prototype.h=function(a){return nt(a,xm)};
nx.prototype.i=function(a,b,c){c=void 0===c?{}:c;b.serializedShareEntity&&(a.serializedSharedEntity=b.serializedShareEntity);c.includeListId&&(a.includeListId=!0)};var ox=new bt("NETWORK_SLI_TOKEN");function px(a){this.h=a}
px.prototype.fetch=function(a,b){var c=this,d,e,f;return x(function(g){c.h&&(d=uc(vc(5,Ic(a,"key")))||"/UNKNOWN_PATH",c.h.start(d));e=b;P("wug_networking_gzip_request")&&(e=Vq(b));f=new window.Request(a,e);return P("web_fetch_promise_cleanup_killswitch")?g.return(Promise.resolve(fetch(f).then(function(h){return c.handleResponse(h)}).catch(function(h){tu(h)}))):g.return(fetch(f).then(function(h){return c.handleResponse(h)}).catch(function(h){tu(h)}))})};
px.prototype.handleResponse=function(a){var b=a.text().then(function(c){return JSON.parse(c.replace(")]}'",""))});
a.redirected||a.ok?this.h&&this.h.success():(this.h&&this.h.vg(),b=b.then(function(c){tu(new lo("Error: API fetch failed",a.status,a.url,c));return Object.assign({},c,{errorMetadata:{status:a.status}})}));
return b};
px[at]=[new ct(ox)];var qx=new bt("NETWORK_MANAGER_TOKEN");var rx;function sx(a){a=void 0===a||a?Cu():Bu();for(var b=[],c=0;c<a.length;c++)b.push("ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_".charAt(a[c]&63));return b.join("")}
;function tx(a){xq.call(this,1,arguments);this.csn=a}
v(tx,xq);var Gq=new yq("screen-created",tx),ux=[],wx=vx,xx=0;function yx(a,b,c,d,e,f,g,h){function l(){tu(new lo("newScreen() parent element does not have a VE - rootVe",b))}
var m=wx(),n=new Hu({veType:b,youtubeData:f,jspbYoutubeData:void 0});f={sequenceGroup:m};e&&(f.cttAuthInfo=e);if(P("il_via_jspb")){e=Xl((new Wl).h(m),n.getAsJspb());c&&c.visualElement?(n=new Ul,c.clientScreenNonce&&G(n,2,c.clientScreenNonce),Vl(n,c.visualElement.getAsJspb()),g&&G(n,4,rm[g]),H(e,Ul,5,n)):c&&l();d&&G(e,3,d);if(P("expectation_logging")&&h&&h.screenCreatedLoggingExpectations){c=new Mk;h=r(h.screenCreatedLoggingExpectations.expectedParentScreens||[]);for(d=h.next();!d.done;d=h.next())d=
d.value,d.screenVeType&&(d=Lk(new Kk,d.screenVeType),le(c,1,Kk,d));H(e,Mk,7,c)}ku(e,f,a)}else e={csn:m,pageVe:n.getAsJson()},P("expectation_logging")&&h&&h.screenCreatedLoggingExpectations&&(e.screenCreatedLoggingExpectations=h.screenCreatedLoggingExpectations),c&&c.visualElement?(e.implicitGesture={parentCsn:c.clientScreenNonce,gesturedVe:c.visualElement.getAsJson()},g&&(e.implicitGesture.gestureType=g)):c&&l(),d&&(e.cloneCsn=d),a?$t("screenCreated",e,a,f):Do("screenCreated",e,f);Dq(Gq,new tx(m));
return m}
function zx(a,b,c,d){var e=d.filter(function(l){l.csn!==b?(l.csn=b,l=!0):l=!1;return l}),f={cttAuthInfo:Ru(b)||void 0,
sequenceGroup:b};d=r(d);for(var g=d.next();!g.done;g=d.next())g=g.value.getAsJson(),(pb(g)||!g.trackingParams&&!g.veType)&&tu(Error("Child VE logged with no data"));if(P("il_via_jspb")){var h=$l((new Yl).h(b),c.getAsJspb());hb(e,function(l){l=l.getAsJspb();le(h,3,Pl,l)});
"UNDEFINED_CSN"===b?Ax("visualElementAttached",f,void 0,h):lu(h,f,a)}else c={csn:b,parentVe:c.getAsJson(),childVes:hb(e,function(l){return l.getAsJson()})},"UNDEFINED_CSN"===b?Ax("visualElementAttached",f,c):a?$t("visualElementAttached",c,a,f):Do("visualElementAttached",c,f)}
function Bx(a,b,c,d,e,f){d={cttAuthInfo:Ru(b)||void 0,sequenceGroup:b};P("il_via_jspb")?(e=(new cm).h(b),c=c.getAsJspb(),c=H(e,Pl,2,c),c=G(c,4,1),f&&H(c,Sl,3,f),"UNDEFINED_CSN"===b?Ax("visualElementShown",d,void 0,c):gu(c,d,a)):(f={csn:b,ve:c.getAsJson(),eventType:1},e&&(f.clientData=e),"UNDEFINED_CSN"===b?Ax("visualElementShown",d,f):a?$t("visualElementShown",f,a,d):Do("visualElementShown",f,d))}
function Cx(a,b,c){var d=!0,e=(d=void 0===d?!1:d)?16:8,f={cttAuthInfo:Ru(b)||void 0,sequenceGroup:b,endOfSequence:d};P("il_via_jspb")?(e=(new bm).h(b),c=c.getAsJspb(),c=H(e,Pl,2,c),G(c,4,d?16:8),"UNDEFINED_CSN"===b?Ax("visualElementHidden",f,void 0,c):hu(c,f,a)):(d={csn:b,ve:c.getAsJson(),eventType:e},"UNDEFINED_CSN"===b?Ax("visualElementHidden",f,d):a?$t("visualElementHidden",d,a,f):Do("visualElementHidden",d,f))}
function vx(){var a;P("enable_web_96_bit_csn")?a=sx():P("enable_web_96_bit_csn_no_crypto")?a=sx(!1):a=od(Kc(Math.random()+""),3);return a}
function Ax(a,b,c,d){ux.push({Mc:a,payload:c,Sa:d,options:b});xx||(xx=Hq())}
function Iq(a){if(ux){for(var b=r(ux),c=b.next();!c.done;c=b.next())if(c=c.value,P("il_via_jspb")&&c.Sa)switch(c.Sa.h(a.csn),c.Mc){case "screenCreated":ku(c.Sa,c.options);break;case "visualElementAttached":lu(c.Sa,c.options);break;case "visualElementShown":gu(c.Sa,c.options);break;case "visualElementHidden":hu(c.Sa,c.options);break;case "visualElementGestured":iu(c.Sa,c.options);break;case "visualElementStateChanged":ju(c.Sa,c.options);break;default:tu(new lo("flushQueue unable to map payloadName to JSPB setter"))}else c.payload&&
(c.payload.csn=a.csn,Do(c.Mc,c.payload,c.options));ux.length=0}xx=0}
;function Dx(){this.j=new Set;this.i=new Set;this.m=new Map;this.client=void 0;this.csn=null}
function Ex(){Dx.h||(Dx.h=new Dx);return Dx.h}
Dx.prototype.l=function(a){this.client=a};
Dx.prototype.h=function(){this.clear();this.csn=Ou()};
Dx.prototype.clear=function(){this.j.clear();this.i.clear();this.m.clear();this.csn=null};function Fx(){}
Fx.prototype.l=function(a){Om(Ex().l).bind(Ex())(a)};
Fx.prototype.clear=function(){Om(Ex().clear).bind(Ex())()};function Gx(){this.s=[];this.v=[];this.h=[];this.ra=[];this.m=[];this.X=[];this.i=new Set;this.H=new Map}
Gx.prototype.l=function(a){this.client=a};
function Hx(){Gx.h||(Gx.h=new Gx);var a=Gx.h;var b=16623;var c=void 0===c?{}:c;Om(function(){Vu.includes(b)||(tu(new lo("createClientScreen() called with a non-page VE",b)),b=83769);c.isHistoryNavigation||(a.ra=[],a.h.push({rootVe:b,key:c.key||""}));a.s=[];a.v=[];c.ld?Ix(a,b,c):Jx(a,b,c)})()}
function Kx(a,b,c){c=void 0===c?0:c;Om(function(){b.then(function(d){a.i.has(c)&&a.j&&a.j();var e=Ou(c),f=Mu(c);if(e&&f){var g;(null==d?0:null==(g=d.response)?0:g.trackingParams)&&zx(a.client,e,f,[Iu(d.response.trackingParams)]);var h;(null==d?0:null==(h=d.playerResponse)?0:h.trackingParams)&&zx(a.client,e,f,[Iu(d.playerResponse.trackingParams)])}})})()}
function Lx(a,b,c,d){d=void 0===d?0:d;Om(function(){if(a.i.has(d))return a.s.push([b,c]),!0;var e=Ou(d),f=c||Mu(d);return e&&f?(zx(a.client,e,f,[b]),!0):!1})()}
Gx.prototype.clickCommand=function(a,b,c){var d=a.clickTrackingParams;c=void 0===c?0:c;if(d)if(c=Ou(void 0===c?0:c)){a=this.client;var e=Iu(d);d={cttAuthInfo:Ru(c)||void 0,sequenceGroup:c};P("il_via_jspb")?(b=(new am).h(c),e=e.getAsJspb(),b=H(b,Pl,2,e),G(b,4,rm.INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK),"UNDEFINED_CSN"===c?Ax("visualElementGestured",d,void 0,b):iu(b,d,a)):(e={csn:c,ve:e.getAsJson(),gestureType:"INTERACTION_LOGGING_GESTURE_TYPE_GENERIC_CLICK"},b&&(e.clientData=b),"UNDEFINED_CSN"===
c?Ax("visualElementGestured",d,e):a?$t("visualElementGestured",e,a,d):Do("visualElementGestured",e,d));b=!0}else b=!1;else b=!1;return b};
Gx.prototype.visualElementStateChanged=function(a,b,c){c=void 0===c?0:c;0===c&&this.i.has(c)?this.v.push([a,b]):Mx(this,a,b,c)};
function Mx(a,b,c,d){d=void 0===d?0:d;var e=Ou(d);d=b||Mu(d);e&&d&&(a=a.client,b={cttAuthInfo:Ru(e)||void 0,sequenceGroup:e},P("il_via_jspb")?(c=new dm,c.h(e),d=d.getAsJspb(),H(c,Pl,2,d),"UNDEFINED_CSN"===e?Ax("visualElementStateChanged",b,void 0,c):ju(c,b,a)):(c={csn:e,ve:d.getAsJson(),clientData:c},"UNDEFINED_CSN"===e?Ax("visualElementStateChanged",b,c):a?$t("visualElementStateChanged",c,a,b):Do("visualElementStateChanged",c,b)))}
function Ix(a,b,c){c=void 0===c?{}:c;a.i.add(c.layer||0);a.j=function(){Jx(a,b,c);var f=Mu(c.layer);if(f){for(var g=r(a.s),h=g.next();!h.done;h=g.next())h=h.value,Lx(a,h[0],h[1]||f,c.layer);f=r(a.v);for(g=f.next();!g.done;g=f.next())g=g.value,Mx(a,g[0],g[1])}};
Ou(c.layer)||a.j();if(c.ld)for(var d=r(c.ld),e=d.next();!e.done;e=d.next())Kx(a,e.value,c.layer);else su(Error("Delayed screen needs a data promise."))}
function Jx(a,b,c){c=void 0===c?{}:c;var d=void 0;c.layer||(c.layer=0);d=void 0!==c.Le?c.Le:c.layer;var e=Ou(d);d=Mu(d);var f;d&&(void 0!==c.parentCsn?f={clientScreenNonce:c.parentCsn,visualElement:d}:e&&"UNDEFINED_CSN"!==e&&(f={clientScreenNonce:e,visualElement:d}));var g,h=O("EVENT_ID");"UNDEFINED_CSN"===e&&h&&(g={servletData:{serializedServletEventId:h}});try{var l=yx(a.client,b,f,c.kd,c.cttAuthInfo,g,c.wg,c.loggingExpectations)}catch(q){wu(q,{Gg:b,rootVe:d,Dg:void 0,sg:e,Cg:f,kd:c.kd});su(q);
return}Su(l,b,c.layer,c.cttAuthInfo);e&&"UNDEFINED_CSN"!==e&&d&&!Pu(e)&&Cx(a.client,e,d);a.h[a.h.length-1]&&!a.h[a.h.length-1].csn&&(a.h[a.h.length-1].csn=l||"");Om(Aw)({clientScreenNonce:l},void 0,!1);Fx.h||(Fx.h=new Fx);Om(Ex().h).bind(Ex())();var m=Mu(c.layer);e&&"UNDEFINED_CSN"!==e&&m&&(P("web_mark_root_visible")||P("music_web_mark_root_visible"))&&Om(Bx)(void 0,l,m,void 0,void 0,void 0);a.i.delete(c.layer||0);a.j=void 0;var n;null==(n=a.H.get(c.layer))||n.forEach(function(q,u){q?Lx(a,u,q,c.layer):
m&&Lx(a,u,m,c.layer)});
Nx(a)}
function Nx(a){for(var b=0;b<a.m.length;b++){var c=a.m[b];try{c()}catch(d){su(d)}}for(b=a.m.length=0;b<a.X.length;b++){c=a.X[b];try{c()}catch(d){su(d)}}}
;function Ox(){var a,b,c;return x(function(d){if(1==d.h)return a=ht().resolve(bx),a?w(d,Xw(a),2):(tu(Error("InnertubeTransportService unavailable in fetchDatasyncIds")),d.return(void 0));if(b=d.i){if(b.errorMetadata)return tu(Error("Datasync IDs fetch responded with "+b.errorMetadata.status+": "+b.error)),d.return(void 0);c=b.tg;return d.return(c)}tu(Error("Network request to get Datasync IDs failed."));return d.return(void 0)})}
;var Px=y.caches,Qx;function Rx(a){var b=a.indexOf(":");return-1===b?{Cd:a}:{Cd:a.substring(0,b),datasyncId:a.substring(b+1)}}
function Sx(){return x(function(a){if(void 0!==Qx)return a.return(Qx);Qx=new Promise(function(b){var c;return x(function(d){switch(d.h){case 1:return xa(d,2),w(d,Px.open("test-only"),4);case 4:return w(d,Px.delete("test-only"),5);case 5:ya(d,3);break;case 2:if(c=za(d),c instanceof Error&&"SecurityError"===c.name)return b(!1),d.return();case 3:b("caches"in window),d.h=0}})});
return a.return(Qx)})}
function Tx(a){var b,c,d,e,f,g,h;x(function(l){if(1==l.h)return w(l,Sx(),2);if(3!=l.h){if(!l.i)return l.return(!1);b=[];return w(l,Px.keys(),3)}c=l.i;d=r(c);for(e=d.next();!e.done;e=d.next())f=e.value,g=Rx(f),h=g.datasyncId,!h||a.includes(h)||b.push(Px.delete(f));return l.return(Promise.all(b).then(function(m){return m.some(function(n){return n})}))})}
function Ux(){var a,b,c,d,e,f,g;return x(function(h){if(1==h.h)return w(h,Sx(),2);if(3!=h.h){if(!h.i)return h.return(!1);a=no("cache contains other");return w(h,Px.keys(),3)}b=h.i;c=r(b);for(d=c.next();!d.done;d=c.next())if(e=d.value,f=Rx(e),(g=f.datasyncId)&&g!==a)return h.return(!0);return h.return(!1)})}
;function Vx(){try{return!!self.localStorage}catch(a){return!1}}
;function Wx(a){a=a.match(/(.*)::.*::.*/);if(null!==a)return a[1]}
function Xx(a){if(Vx()){var b=Object.keys(window.localStorage);b=r(b);for(var c=b.next();!c.done;c=b.next()){c=c.value;var d=Wx(c);void 0===d||a.includes(d)||self.localStorage.removeItem(c)}}}
function Yx(){if(!Vx())return!1;var a=no(),b=Object.keys(window.localStorage);b=r(b);for(var c=b.next();!c.done;c=b.next())if(c=Wx(c.value),void 0!==c&&c!==a)return!0;return!1}
;function Zx(){Ox().then(function(a){a&&(Qp(a),Tx(a),Xx(a))})}
function $x(){var a=new os;Di.ea(function(){var b,c,d,e;return x(function(f){switch(f.h){case 1:if(P("ytidb_clear_optimizations_killswitch")){f.A(2);break}b=no("clear");if(b.startsWith("V")){var g=[b];Qp(g);Tx(g);Xx(g);return f.return()}c=Yx();return w(f,Ux(),3);case 3:return d=f.i,w(f,Rp(),4);case 4:if(e=f.i,!c&&!d&&!e)return f.return();case 2:a.oa()?Zx():a.l.add("publicytnetworkstatus-online",Zx,!0,void 0,void 0),f.h=0}})})}
;var ii=fa(["data-"]);function ay(a){a&&(a.dataset?a.dataset[by("loaded")]="true":hi(a))}
function cy(a,b){return a?a.dataset?a.dataset[by(b)]:a.getAttribute("data-"+b):null}
var dy={};function by(a){return dy[a]||(dy[a]=String(a).replace(/\-([a-z])/g,function(b,c){return c.toUpperCase()}))}
;var ey=/\.vflset|-vfl[a-zA-Z0-9_+=-]+/,fy=/-[a-zA-Z]{2,3}_[a-zA-Z]{2,3}(?=(\/|$))/;function gy(a,b,c){c=void 0===c?null:c;if(window.spf&&spf.script){c="";if(a){var d=a.indexOf("jsbin/"),e=a.lastIndexOf(".js"),f=d+6;-1<d&&-1<e&&e>f&&(c=a.substring(f,e),c=c.replace(ey,""),c=c.replace(fy,""),c=c.replace("debug-",""),c=c.replace("tracing-",""))}spf.script.load(a,c,b)}else hy(a,b,c)}
function hy(a,b,c){c=void 0===c?null:c;var d=iy(a),e=document.getElementById(d),f=e&&cy(e,"loaded"),g=e&&!f;f?b&&b():(b&&(f=Vs(d,b),b=""+Ra(b),jy[b]=f),g||(e=ky(a,d,function(){cy(e,"loaded")||(ay(e),Ys(d),kn(Xa(Zs,d),0))},c)))}
function ky(a,b,c,d){d=void 0===d?null:d;var e=Cf("SCRIPT");e.id=b;e.onload=function(){c&&setTimeout(c,0)};
e.onreadystatechange=function(){switch(e.readyState){case "loaded":case "complete":e.onload()}};
d&&e.setAttribute("nonce",d);ki(e,Gk(a));a=document.getElementsByTagName("head")[0]||document.body;a.insertBefore(e,a.firstChild);return e}
function ly(a){a=iy(a);var b=document.getElementById(a);b&&(Zs(a),b.parentNode.removeChild(b))}
function my(a,b){a&&b&&(a=""+Ra(b),(a=jy[a])&&Xs(a))}
function iy(a){var b=document.createElement("a");mc(b,a);a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"js-"+sc(a)}
var jy={};var ny=[],oy=!1;function py(){if(!P("disable_biscotti_fetch_for_ad_blocker_detection")&&!P("disable_biscotti_fetch_entirely_for_all_web_clients")&&dv()){var a=O("PLAYER_VARS",{});if("1"!=rb(a)&&!ev(a)){var b=function(){oy=!0;"google_ad_status"in window?Km("DCLKSTAT",1):Km("DCLKSTAT",2)};
try{gy("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}ny.push(Di.ea(function(){if(!(oy||"google_ad_status"in window)){try{my("//static.doubleclick.net/instream/ad_status.js",b)}catch(c){}oy=!0;Km("DCLKSTAT",3)}},5E3))}}}
function qy(){var a=Number(O("DCLKSTAT",0));return isNaN(a)?0:a}
;var Cy=window,Dy,Ey=P("web_enable_lifecycle_monitoring")&&(null==(Dy=Cy.performance)?void 0:Dy.measure);function Fy(a){var b=this;var c=void 0===c?0:c;var d=void 0===d?to():d;this.l=c;this.j=d;this.i=new $h;this.h=a;for(a={Za:0};a.Za<this.h.length;a={Eb:a.Eb,Za:a.Za},a.Za++)a.Eb=this.h[a.Za],c=function(e){return function(){e.Eb.Fc();b.h[e.Za].nc=!0;b.h.every(function(f){return!0===f.nc})&&b.i.resolve()}}(a),d=qo(c,Gy(this,a.Eb)),this.h[a.Za]=Object.assign({},a.Eb,{Fc:c,
jobId:d})}
function Hy(a){var b=Array.from(a.h.keys()).sort(function(d,e){return Gy(a,a.h[e])-Gy(a,a.h[d])});
b=r(b);for(var c=b.next();!c.done;c=b.next())c=a.h[c.value],void 0===c.jobId||c.nc||(a.j.Ba(c.jobId),qo(c.Fc,10))}
Fy.prototype.cancel=function(){for(var a=r(this.h),b=a.next();!b.done;b=a.next())b=b.value,void 0===b.jobId||b.nc||this.j.Ba(b.jobId),b.nc=!0;this.i.resolve()};
function Gy(a,b){var c;return null!=(c=b.priority)?c:a.l}
;function Iy(a){this.state=a;this.plugins=[];this.s=void 0;this.v={};Ey&&window.performance.mark(this.state+"-start")}
k=Iy.prototype;k.install=function(a){this.plugins.push(a);return this};
k.uninstall=function(){var a=this;Ja.apply(0,arguments).forEach(function(b){b=a.plugins.indexOf(b);-1<b&&a.plugins.splice(b,1)})};
k.transition=function(a,b){var c=this;Jy(this);var d=this.transitions.find(function(f){return Array.isArray(f.from)?f.from.find(function(g){return g===c.state&&f.K===a}):f.from===c.state&&f.K===a});
if(d){this.j&&(Hy(this.j),this.j=void 0);Ky(this,a,b);this.state=a;Ey&&window.performance.mark(this.state+"-start");d=d.action.bind(this);var e=this.plugins.filter(function(f){return f[a]}).map(function(f){return f[a]});
d(Ly(this,e),b)}else throw Error("no transition specified from "+this.state+" to "+a);};
function Ly(a,b){var c=b.filter(function(e){return 10===My(a,e)}),d=b.filter(function(e){return 10!==My(a,e)});
return a.v.Kg?function(){var e=Ja.apply(0,arguments);return x(function(f){if(1==f.h)return w(f,a.Se.apply(a,[c].concat(ha(e))),2);a.Ld.apply(a,[d].concat(ha(e)));f.h=0})}:function(){var e=Ja.apply(0,arguments);
a.Te.apply(a,[c].concat(ha(e)));a.Ld.apply(a,[d].concat(ha(e)))}}
k.Te=function(a){var b=Ja.apply(1,arguments);to();for(var c={},d=r(a),e=d.next();!e.done;c={ob:c.ob},e=d.next())c.ob=e.value,ro(function(f){return function(){Ny(f.ob.name);f.ob.callback.apply(f.ob,ha(b));Oy(f.ob.name)}}(c))};
k.Se=function(a){var b=Ja.apply(1,arguments),c,d,e,f;return x(function(g){1==g.h&&(to(),c={},d=r(a),e=d.next());if(3!=g.h){if(e.done)return g.A(0);c.ab=e.value;c.Cb=void 0;f=function(h){return function(){Ny(h.ab.name);var l=h.ab.callback.apply(h.ab,ha(b));l&&(l instanceof Pf||l instanceof Promise)?h.Cb=l.then(function(){Oy(h.ab.name)}):Oy(h.ab.name)}}(c);
ro(f);return c.Cb?w(g,c.Cb,3):g.A(3)}c={ab:c.ab,Cb:c.Cb};e=d.next();return g.A(2)})};
k.Ld=function(a){var b=Ja.apply(1,arguments),c=this,d=a.map(function(e){return{Fc:function(){Ny(e.name);e.callback.apply(e,ha(b));Oy(e.name)},
priority:My(c,e)}});
d.length&&(this.j=new Fy(d))};
function My(a,b){var c,d;return null!=(d=null!=(c=a.s)?c:b.priority)?d:0}
function Jy(a){if(Ey){var b=a.state+"-start",c=a.state+"-end";window.performance.mark(c);window.performance.measure(a.state,b,c)}}
function Ny(a){Ey&&a&&window.performance.mark(a+"-start")}
function Oy(a){if(Ey&&a){var b=a+"-start",c=a+"-end";window.performance.mark(c);window.performance.measure(a,b,c)}}
function Ky(a,b,c){Ey&&(console.groupCollapsed("["+a.constructor.name+"] '"+a.state+"' to '"+b+"'"),console.log("with message: ",c),console.groupEnd())}
da.Object.defineProperties(Iy.prototype,{currentState:{configurable:!0,enumerable:!0,get:function(){return this.state}}});function Py(a){Iy.call(this,void 0===a?"document_active":a);var b=this;this.s=10;this.h=new Map;this.transitions=[{from:"document_active",K:"document_disposed_preventable",action:this.X},{from:"document_active",K:"document_disposed",action:this.l},{from:"document_disposed_preventable",K:"document_disposed",action:this.l},{from:"document_disposed_preventable",K:"flush_logs",action:this.m},{from:"document_disposed_preventable",K:"document_active",action:this.i},{from:"document_disposed",K:"flush_logs",
action:this.m},{from:"document_disposed",K:"document_active",action:this.i},{from:"document_disposed",K:"document_disposed",action:function(){}},
{from:"flush_logs",K:"document_active",action:this.i}];window.addEventListener("pagehide",function(c){b.transition("document_disposed",{event:c})});
window.addEventListener("beforeunload",function(c){b.transition("document_disposed_preventable",{event:c})})}
v(Py,Iy);Py.prototype.X=function(a,b){if(!this.h.get("document_disposed_preventable")){a(null==b?void 0:b.event);var c,d;if((null==b?0:null==(c=b.event)?0:c.defaultPrevented)||(null==b?0:null==(d=b.event)?0:d.returnValue)){b.event.returnValue||(b.event.returnValue=!0);b.event.defaultPrevented||b.event.preventDefault();this.h=new Map;this.transition("document_active");return}}this.h.set("document_disposed_preventable",!0);this.h.get("document_disposed")?this.transition("flush_logs"):this.transition("document_disposed")};
Py.prototype.l=function(a,b){this.h.get("document_disposed")?this.transition("document_active"):(a(null==b?void 0:b.event),this.h.set("document_disposed",!0),this.transition("flush_logs"))};
Py.prototype.m=function(a,b){a(null==b?void 0:b.event);this.transition("document_active")};
Py.prototype.i=function(){this.h=new Map};function Qy(a){Iy.call(this,void 0===a?"document_visibility_unknown":a);var b=this;this.transitions=[{from:"document_visibility_unknown",K:"document_visible",action:this.i},{from:"document_visibility_unknown",K:"document_hidden",action:this.h},{from:"document_visibility_unknown",K:"document_foregrounded",action:this.m},{from:"document_visibility_unknown",K:"document_backgrounded",action:this.l},{from:"document_visible",K:"document_hidden",action:this.h},{from:"document_visible",K:"document_foregrounded",
action:this.m},{from:"document_visible",K:"document_visible",action:this.i},{from:"document_foregrounded",K:"document_visible",action:this.i},{from:"document_foregrounded",K:"document_hidden",action:this.h},{from:"document_foregrounded",K:"document_foregrounded",action:this.m},{from:"document_hidden",K:"document_visible",action:this.i},{from:"document_hidden",K:"document_backgrounded",action:this.l},{from:"document_hidden",K:"document_hidden",action:this.h},{from:"document_backgrounded",K:"document_hidden",
action:this.h},{from:"document_backgrounded",K:"document_backgrounded",action:this.l},{from:"document_backgrounded",K:"document_visible",action:this.i}];document.addEventListener("visibilitychange",function(c){"visible"===document.visibilityState?b.transition("document_visible",{event:c}):b.transition("document_hidden",{event:c})});
P("visibility_lifecycles_dynamic_backgrounding")&&(window.addEventListener("blur",function(c){b.transition("document_backgrounded",{event:c})}),window.addEventListener("focus",function(c){b.transition("document_foregrounded",{event:c})}))}
v(Qy,Iy);Qy.prototype.i=function(a,b){a(null==b?void 0:b.event);P("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_foregrounded")};
Qy.prototype.h=function(a,b){a(null==b?void 0:b.event);P("visibility_lifecycles_dynamic_backgrounding")&&this.transition("document_backgrounded")};
Qy.prototype.l=function(a,b){a(null==b?void 0:b.event)};
Qy.prototype.m=function(a,b){a(null==b?void 0:b.event)};function Ry(){this.j=new Py;this.l=new Qy}
Ry.prototype.install=function(){var a=Ja.apply(0,arguments),b=this;a.forEach(function(c){b.j.install(c)});
a.forEach(function(c){b.l.install(c)})};function Sy(){Ry.call(this);var a={};this.install((a.document_disposed={callback:this.h},a));a={};this.install((a.flush_logs={callback:this.i},a))}
v(Sy,Ry);Sy.prototype.i=function(){if(P("web_fp_via_jspb")){var a=new Ol,b=Ou();b&&G(a,1,b);b=new pm;je(b,Ol,380,qm,a);fu(b);P("web_fp_via_jspb_and_json")&&Do("finalPayload",{csn:Ou()})}else Do("finalPayload",{csn:Ou()})};
Sy.prototype.h=function(){yu(zu)};function Ty(){}
function Uy(){var a=B("ytglobal.storage_");a||(a=new Ty,z("ytglobal.storage_",a));return a}
Ty.prototype.estimate=function(){var a,b,c;return x(function(d){a=navigator;return(null==(b=a.storage)?0:b.estimate)?d.return(a.storage.estimate()):(null==(c=a.webkitTemporaryStorage)?0:c.queryUsageAndQuota)?d.return(Vy()):d.return()})};
function Vy(){var a=navigator;return new Promise(function(b,c){var d;null!=(d=a.webkitTemporaryStorage)&&d.queryUsageAndQuota?a.webkitTemporaryStorage.queryUsageAndQuota(function(e,f){b({usage:e,quota:f})},function(e){c(e)}):c(Error("webkitTemporaryStorage is not supported."))})}
z("ytglobal.storageClass_",Ty);function Bo(a,b){var c=this;this.handleError=a;this.h=b;this.i=!1;void 0===self.document||self.addEventListener("beforeunload",function(){c.i=!0});
this.j=Math.random()<=dn("ytidb_transaction_ended_event_rate_limit_session",.2)}
Bo.prototype.logEvent=function(a,b){switch(a){case "IDB_DATA_CORRUPTED":P("idb_data_corrupted_killswitch")||this.h("idbDataCorrupted",b);break;case "IDB_UNEXPECTEDLY_CLOSED":this.h("idbUnexpectedlyClosed",b);break;case "IS_SUPPORTED_COMPLETED":P("idb_is_supported_completed_killswitch")||this.h("idbIsSupportedCompleted",b);break;case "QUOTA_EXCEEDED":Wy(this,b);break;case "TRANSACTION_ENDED":this.j&&Math.random()<=dn("ytidb_transaction_ended_event_rate_limit_transaction",.1)&&this.h("idbTransactionEnded",
b);break;case "TRANSACTION_UNEXPECTEDLY_ABORTED":a=Object.assign({},b,{hasWindowUnloaded:this.i}),this.h("idbTransactionAborted",a)}};
function Wy(a,b){Uy().estimate().then(function(c){c=Object.assign({},b,{isSw:void 0===self.document,isIframe:self!==self.top,deviceStorageUsageMbytes:Xy(null==c?void 0:c.usage),deviceStorageQuotaMbytes:Xy(null==c?void 0:c.quota)});a.h("idbQuotaExceeded",c)})}
function Xy(a){return"undefined"===typeof a?"-1":String(Math.ceil(a/1048576))}
;function Yy(a,b,c){K.call(this);var d=this;c=c||O("POST_MESSAGE_ORIGIN")||window.document.location.protocol+"//"+window.document.location.hostname;this.l=b||null;this.targetOrigin="*";this.m=c;this.sessionId=null;this.i="widget";this.M=!!a;this.H=function(e){a:if(!("*"!=d.m&&e.origin!=d.m||d.l&&e.source!=d.l||"string"!==typeof e.data)){try{var f=JSON.parse(e.data)}catch(g){break a}if(!(null==f||d.M&&(d.sessionId&&d.sessionId!=f.id||d.i&&d.i!=f.channel))&&f)switch(f.event){case "listening":"null"!=
e.origin&&(d.m=d.targetOrigin=e.origin);d.l=e.source;d.sessionId=f.id;d.j&&(d.j(),d.j=null);break;case "command":d.s&&(!d.v||0<=eb(d.v,f.func))&&d.s(f.func,f.args,e.origin)}}};
this.v=this.j=this.s=null;window.addEventListener("message",this.H)}
v(Yy,K);Yy.prototype.sendMessage=function(a,b){if(b=b||this.l){this.sessionId&&(a.id=this.sessionId);this.i&&(a.channel=this.i);try{var c=JSON.stringify(a);b.postMessage(c,this.targetOrigin)}catch(d){Qm(d)}}};
Yy.prototype.L=function(){window.removeEventListener("message",this.H);K.prototype.L.call(this)};function Zy(){this.i=[];this.isReady=!1;this.j={};var a=this.h=new Yy(!!O("WIDGET_ID_ENFORCE")),b=this.Ne.bind(this);a.s=b;a.v=null;this.h.i="widget";if(a=O("WIDGET_ID"))this.h.sessionId=a}
k=Zy.prototype;k.Ne=function(a,b,c){"addEventListener"===a&&b?this.Ec(b[0],c):this.Vc(a,b,c)};
k.Vc=function(){};
k.xc=function(a){var b=this;return function(c){return b.sendMessage(a,c)}};
k.Ec=function(a,b){this.j[a]||"onReady"===a||(this.addEventListener(a,this.xc(a,b)),this.j[a]=!0)};
k.addEventListener=function(){};
k.qe=function(){this.isReady=!0;this.sendMessage("initialDelivery",this.Ac());this.sendMessage("onReady");fb(this.i,this.Jd,this);this.i=[]};
k.Ac=function(){return null};
function $y(a,b){a.sendMessage("infoDelivery",b)}
k.Jd=function(a){this.isReady?this.h.sendMessage(a):this.i.push(a)};
k.sendMessage=function(a,b){this.Jd({event:a,info:void 0===b?null:b})};
k.dispose=function(){this.h=null};var az={},bz=(az["api.invalidparam"]=2,az.auth=150,az["drm.auth"]=150,az["heartbeat.net"]=150,az["heartbeat.servererror"]=150,az["heartbeat.stop"]=150,az["html5.unsupportedads"]=5,az["fmt.noneavailable"]=5,az["fmt.decode"]=5,az["fmt.unplayable"]=5,az["html5.missingapi"]=5,az["html5.unsupportedlive"]=5,az["drm.unavailable"]=5,az["mrm.blocked"]=151,az);var cz=new Set("endSeconds startSeconds mediaContentUrl suggestedQuality videoId rct rctn".split(" "));function dz(a){return(0===a.search("cue")||0===a.search("load"))&&"loadModule"!==a}
function ez(a,b,c){if("string"===typeof a)return{videoId:a,startSeconds:b,suggestedQuality:c};b={};c=r(cz);for(var d=c.next();!d.done;d=c.next())d=d.value,a[d]&&(b[d]=a[d]);return b}
function fz(a,b,c,d){if(Pa(a)&&!Array.isArray(a)){b="playlist list listType index startSeconds suggestedQuality".split(" ");c={};for(d=0;d<b.length;d++){var e=b[d];a[e]&&(c[e]=a[e])}return c}b={index:b,startSeconds:c,suggestedQuality:d};"string"===typeof a&&16===a.length?b.list="PL"+a:b.playlist=a;return b}
;function gz(a){Zy.call(this);this.listeners=[];this.l=!1;this.api=a;this.addEventListener("onReady",this.onReady.bind(this));this.addEventListener("onVideoProgress",this.Ze.bind(this));this.addEventListener("onVolumeChange",this.af.bind(this));this.addEventListener("onApiChange",this.Ue.bind(this));this.addEventListener("onPlaybackQualityChange",this.We.bind(this));this.addEventListener("onPlaybackRateChange",this.Xe.bind(this));this.addEventListener("onStateChange",this.Ye.bind(this));this.addEventListener("onWebglSettingsChanged",
this.bf.bind(this))}
v(gz,Zy);k=gz.prototype;
k.Vc=function(a,b,c){if(this.api.isExternalMethodAvailable(a,c)){b=b||[];if(0<b.length&&dz(a)){var d=b;if(Pa(d[0])&&!Array.isArray(d[0]))var e=d[0];else switch(e={},a){case "loadVideoById":case "cueVideoById":e=ez(d[0],void 0!==d[1]?Number(d[1]):void 0,d[2]);break;case "loadVideoByUrl":case "cueVideoByUrl":e=d[0];"string"===typeof e&&(e={mediaContentUrl:e,startSeconds:void 0!==d[1]?Number(d[1]):void 0,suggestedQuality:d[2]});b:{if((d=e.mediaContentUrl)&&(d=/\/([ve]|embed)\/([^#?]+)/.exec(d))&&d[2]){d=
d[2];break b}d=null}e.videoId=d;e=ez(e);break;case "loadPlaylist":case "cuePlaylist":e=fz(d[0],d[1],d[2],d[3])}b.length=1;b[0]=e}this.api.handleExternalCall(a,b,c);dz(a)&&$y(this,this.Ac())}};
k.Ec=function(a,b){"onReady"===a?this.api.logApiCall(a+" invocation",b):"onError"===a&&this.l&&(this.api.logApiCall(a+" invocation",b,this.errorCode),this.errorCode=void 0);this.api.logApiCall(a+" registration",b);Zy.prototype.Ec.call(this,a,b)};
k.xc=function(a,b){var c=this,d=Zy.prototype.xc.call(this,a,b);return function(e){"onError"===a?c.api.logApiCall(a+" invocation",b,e):c.api.logApiCall(a+" invocation",b);d(e)}};
k.onReady=function(){var a=this.qe.bind(this);this.h.j=a;a=this.api.getVideoData();if(!a.isPlayable){this.l=!0;a=a.errorCode;var b=void 0===b?5:b;this.errorCode=a?bz[a]||b:b;this.sendMessage("onError",this.errorCode.toString())}};
k.addEventListener=function(a,b){this.listeners.push({eventType:a,listener:b});this.api.addEventListener(a,b)};
k.Ac=function(){if(!this.api)return null;var a=this.api.getApiInterface();kb(a,"getVideoData");for(var b={apiInterface:a},c=0,d=a.length;c<d;c++){var e=a[c];if(0===e.search("get")||0===e.search("is")){var f=0;0===e.search("get")?f=3:0===e.search("is")&&(f=2);f=e.charAt(f).toLowerCase()+e.substr(f+1);try{var g=this.api[e]();b[f]=g}catch(h){}}}b.videoData=this.api.getVideoData();b.currentTimeLastUpdated_=Date.now()/1E3;return b};
k.Ye=function(a){a={playerState:a,currentTime:this.api.getCurrentTime(),duration:this.api.getDuration(),videoData:this.api.getVideoData(),videoStartBytes:0,videoBytesTotal:this.api.getVideoBytesTotal(),videoLoadedFraction:this.api.getVideoLoadedFraction(),playbackQuality:this.api.getPlaybackQuality(),availableQualityLevels:this.api.getAvailableQualityLevels(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getVideoUrl&&
(a.videoUrl=this.api.getVideoUrl());this.api.getVideoContentRect&&(a.videoContentRect=this.api.getVideoContentRect());this.api.getProgressState&&(a.progressState=this.api.getProgressState());this.api.getPlaylist&&(a.playlist=this.api.getPlaylist());this.api.getPlaylistIndex&&(a.playlistIndex=this.api.getPlaylistIndex());this.api.getStoryboardFormat&&(a.storyboardFormat=this.api.getStoryboardFormat());$y(this,a)};
k.We=function(a){$y(this,{playbackQuality:a})};
k.Xe=function(a){$y(this,{playbackRate:a})};
k.Ue=function(){for(var a=this.api.getOptions(),b={namespaces:a},c=0,d=a.length;c<d;c++){var e=a[c],f=this.api.getOptions(e);b[e]={options:f};for(var g=0,h=f.length;g<h;g++){var l=f[g],m=this.api.getOption(e,l);b[e][l]=m}}this.sendMessage("apiInfoDelivery",b)};
k.af=function(){$y(this,{muted:this.api.isMuted(),volume:this.api.getVolume()})};
k.Ze=function(a){a={currentTime:a,videoBytesLoaded:this.api.getVideoBytesLoaded(),videoLoadedFraction:this.api.getVideoLoadedFraction(),currentTimeLastUpdated_:Date.now()/1E3,playbackRate:this.api.getPlaybackRate(),mediaReferenceTime:this.api.getMediaReferenceTime()};this.api.getProgressState&&(a.progressState=this.api.getProgressState());$y(this,a)};
k.bf=function(){var a={sphericalProperties:this.api.getSphericalProperties()};$y(this,a)};
k.dispose=function(){Zy.prototype.dispose.call(this);for(var a=0;a<this.listeners.length;a++){var b=this.listeners[a];this.api.removeEventListener(b.eventType,b.listener)}this.listeners=[]};function hz(a){K.call(this);this.i={};this.started=!1;this.connection=a;this.connection.subscribe("command",this.Fd,this)}
v(hz,K);k=hz.prototype;k.start=function(){this.started||this.h()||(this.started=!0,this.connection.lb("RECEIVING"))};
k.lb=function(a,b){this.started&&!this.h()&&this.connection.lb(a,b)};
k.Fd=function(a,b,c){if(this.started&&!this.h()){var d=b||{};switch(a){case "addEventListener":"string"===typeof d.event&&this.addListener(d.event);break;case "removeEventListener":"string"===typeof d.event&&this.removeListener(d.event);break;default:this.api.isReady()&&this.api.isExternalMethodAvailable(a,c||null)&&(b=iz(a,b||{}),c=this.api.handleExternalCall(a,b,c||null),(c=jz(a,c))&&this.lb(a,c))}}};
k.addListener=function(a){if(!(a in this.i)){var b=this.Ve.bind(this,a);this.i[a]=b;this.addEventListener(a,b)}};
k.Ve=function(a,b){this.started&&!this.h()&&this.connection.lb(a,this.zc(a,b))};
k.zc=function(a,b){if(null!=b)return{value:b}};
k.removeListener=function(a){a in this.i&&(this.removeEventListener(a,this.i[a]),delete this.i[a])};
k.L=function(){var a=this.connection;a.h()||aj(a.i,"command",this.Fd,this);this.connection=null;for(var b in this.i)this.i.hasOwnProperty(b)&&this.removeListener(b);K.prototype.L.call(this)};function kz(a,b){hz.call(this,b);this.api=a;this.start()}
v(kz,hz);kz.prototype.addEventListener=function(a,b){this.api.addEventListener(a,b)};
kz.prototype.removeEventListener=function(a,b){this.api.removeEventListener(a,b)};
function iz(a,b){switch(a){case "loadVideoById":return a=ez(b),[a];case "cueVideoById":return a=ez(b),[a];case "loadVideoByPlayerVars":return[b];case "cueVideoByPlayerVars":return[b];case "loadPlaylist":return a=fz(b),[a];case "cuePlaylist":return a=fz(b),[a];case "seekTo":return[b.seconds,b.allowSeekAhead];case "playVideoAt":return[b.index];case "setVolume":return[b.volume];case "setPlaybackQuality":return[b.suggestedQuality];case "setPlaybackRate":return[b.suggestedRate];case "setLoop":return[b.loopPlaylists];
case "setShuffle":return[b.shufflePlaylist];case "getOptions":return[b.module];case "getOption":return[b.module,b.option];case "setOption":return[b.module,b.option,b.value];case "handleGlobalKeyDown":return[b.keyCode,b.shiftKey,b.ctrlKey,b.altKey,b.metaKey,b.key,b.code]}return[]}
function jz(a,b){switch(a){case "isMuted":return{muted:b};case "getVolume":return{volume:b};case "getPlaybackRate":return{playbackRate:b};case "getAvailablePlaybackRates":return{availablePlaybackRates:b};case "getVideoLoadedFraction":return{videoLoadedFraction:b};case "getPlayerState":return{playerState:b};case "getCurrentTime":return{currentTime:b};case "getPlaybackQuality":return{playbackQuality:b};case "getAvailableQualityLevels":return{availableQualityLevels:b};case "getDuration":return{duration:b};
case "getVideoUrl":return{videoUrl:b};case "getVideoEmbedCode":return{videoEmbedCode:b};case "getPlaylist":return{playlist:b};case "getPlaylistIndex":return{playlistIndex:b};case "getOptions":return{options:b};case "getOption":return{option:b}}}
kz.prototype.zc=function(a,b){switch(a){case "onReady":return;case "onStateChange":return{playerState:b};case "onPlaybackQualityChange":return{playbackQuality:b};case "onPlaybackRateChange":return{playbackRate:b};case "onError":return{errorCode:b}}return hz.prototype.zc.call(this,a,b)};
kz.prototype.L=function(){hz.prototype.L.call(this);delete this.api};function lz(a){a=void 0===a?!1:a;K.call(this);this.i=new $i(a);Ke(this,this.i)}
Ya(lz,K);lz.prototype.subscribe=function(a,b,c){return this.h()?0:this.i.subscribe(a,b,c)};
lz.prototype.m=function(a,b){this.h()||this.i.cb.apply(this.i,arguments)};function mz(a,b,c){lz.call(this);this.l=a;this.j=b;this.id=c}
v(mz,lz);mz.prototype.lb=function(a,b){this.h()||this.l.lb(this.j,this.id,a,b)};
mz.prototype.L=function(){this.j=this.l=null;lz.prototype.L.call(this)};function nz(a,b,c){K.call(this);this.i=a;this.origin=c;this.j=Js(window,"message",this.l.bind(this));this.connection=new mz(this,a,b);Ke(this,this.connection)}
v(nz,K);nz.prototype.lb=function(a,b,c,d){this.h()||a!==this.i||(a={id:b,command:c},d&&(a.data=d),this.i.postMessage(JSON.stringify(a),this.origin))};
nz.prototype.l=function(a){if(!this.h()&&a.origin===this.origin){var b=a.data;if("string"===typeof b){try{b=JSON.parse(b)}catch(d){return}if(b.command){var c=this.connection;c.h()||c.m("command",b.command,b.data,a.origin)}}}};
nz.prototype.L=function(){Ks(this.j);this.i=null;K.prototype.L.call(this)};function oz(){this.state=1;this.h=null}
k=oz.prototype;k.initialize=function(a,b,c){if(a.program){var d,e=null!=(d=a.interpreterUrl)?d:null;if(a.interpreterSafeScript){d=a.interpreterSafeScript.privateDoNotAccessOrElseSafeScriptWrappedValue||"";var f=yb();d=f?f.createScript(d):d;d=new Db(d,Cb)}else d=null!=(f=a.interpreterScript)?f:null;a.interpreterSafeUrl&&(e=a.interpreterSafeUrl,Bb("From proto message. b/166824318"),e=Hb(e.privateDoNotAccessOrElseTrustedResourceUrlWrappedValue||"").toString());pz(this,d,e,a.program,b,c)}else tu(Error("Cannot initialize botguard without program"))};
function pz(a,b,c,d,e,f){var g=void 0===g?"trayride":g;c?(a.state=2,gy(c,function(){window[g]?qz(a,d,g,e):(a.state=3,ly(c),tu(new lo("Unable to load Botguard","from "+c)))},f)):b?(f=Cf("SCRIPT"),b instanceof Db?(b instanceof Db&&b.constructor===Db?b=b.j:(Na(b),b="type_error:SafeScript"),f.textContent=b,ji(f)):f.textContent=b,f.nonce=qc(),document.head.appendChild(f),document.head.removeChild(f),window[g]?qz(a,d,g,e):(a.state=4,tu(new lo("Unable to load Botguard from JS")))):tu(new lo("Unable to load VM; no url or JS provided"))}
function qz(a,b,c,d){a.state=5;try{var e=new ai({program:b,ue:c,Oe:P("att_web_record_metrics")});e.df.then(function(){a.state=6;d&&d(b)});
a.Qc(e)}catch(f){a.state=7,f instanceof Error&&tu(f)}}
k.invoke=function(a){a=void 0===a?{}:a;return this.Tc()?this.Sd({jd:a}):null};
k.dispose=function(){this.Wc()};
k.Wc=function(){this.Qc(null);this.state=8};
k.Tc=function(){return!!this.h};
k.Sd=function(a){return this.h.Md(a)};
k.Qc=function(a){Ie(this.h);this.h=a};function rz(){var a=B("yt.abuse.playerAttLoader");return a&&["bgvma","bgvmb","bgvmc"].every(function(b){return b in a})?a:null}
;function sz(){oz.apply(this,arguments)}
v(sz,oz);sz.prototype.Wc=function(){this.state=8};
sz.prototype.Qc=function(a){var b;null==(b=rz())||b.bgvma();a?(b={bgvma:a.dispose.bind(a),bgvmb:a.snapshot.bind(a),bgvmc:a.Md.bind(a)},z("yt.abuse.playerAttLoader",b),z("yt.abuse.playerAttLoaderRun",function(c){return a.snapshot(c)})):(z("yt.abuse.playerAttLoader",null),z("yt.abuse.playerAttLoaderRun",null))};
sz.prototype.Tc=function(){return!!rz()};
sz.prototype.Sd=function(a){return rz().bgvmc(a)};var tz=new sz;function uz(){return tz.Tc()}
function vz(a){a=void 0===a?{}:a;return tz.invoke(a)}
;function wz(a){a=a||{};var b={},c={};this.url=a.url||"";this.args=a.args||tb(b);this.assets=a.assets||{};this.attrs=a.attrs||tb(c);this.fallback=a.fallback||null;this.fallbackMessage=a.fallbackMessage||null;this.html5=!!a.html5;this.disable=a.disable||{};this.loaded=!!a.loaded;this.messages=a.messages||{}}
wz.prototype.clone=function(){var a=new wz,b;for(b in this)if(this.hasOwnProperty(b)){var c=this[b];"object"==Na(c)?a[b]=tb(c):a[b]=c}return a};var xz=/cssbin\/(?:debug-)?([a-zA-Z0-9_-]+?)(?:-2x|-web|-rtl|-vfl|.css)/;function yz(a){a=a||"";if(window.spf){var b=a.match(xz);spf.style.load(a,b?b[1]:"",void 0)}else zz(a)}
function zz(a){var b=Az(a),c=document.getElementById(b),d=c&&cy(c,"loaded");d||c&&!d||(c=Bz(a,b,function(){cy(c,"loaded")||(ay(c),Ys(b),kn(Xa(Zs,b),0))}))}
function Bz(a,b,c){var d=document.createElement("link");d.id=b;d.onload=function(){c&&setTimeout(c,0)};
a=Gk(a);nc(d,a);(document.getElementsByTagName("head")[0]||document.body).appendChild(d);return d}
function Az(a){var b=Cf("A");mc(b,new Kb(a,Lb));a=b.href.replace(/^[a-zA-Z]+:\/\//,"//");return"css-"+sc(a)}
;function Cz(){K.call(this);this.i=[]}
v(Cz,K);Cz.prototype.L=function(){for(;this.i.length;){var a=this.i.pop();a.target.removeEventListener(a.name,a.callback,void 0)}K.prototype.L.call(this)};function Dz(){Cz.apply(this,arguments)}
v(Dz,Cz);function Ez(a,b,c,d,e){K.call(this);var f=this;this.v=b;this.webPlayerContextConfig=d;this.Vb=e;this.Oa=!1;this.api={};this.ha=this.s=null;this.T=new $i;this.i={};this.Y=this.ia=this.elementId=this.eb=this.config=null;this.V=!1;this.l=this.H=null;this.ma={};this.Wb=["onReady"];this.lastError=null;this.qb=NaN;this.M={};this.Xb=new Dz(this);this.aa=0;this.j=this.m=a;Ke(this,this.T);Fz(this);Gz(this);Ke(this,this.Xb);c?this.aa=kn(function(){f.loadNewVideoConfig(c)},0):d&&(Hz(this),Iz(this))}
v(Ez,K);k=Ez.prototype;k.getId=function(){return this.v};
k.loadNewVideoConfig=function(a){if(!this.h()){this.aa&&(window.clearTimeout(this.aa),this.aa=0);var b=a||{};b instanceof wz||(b=new wz(b));this.config=b;this.setConfig(a);Iz(this);this.isReady()&&Jz(this)}};
function Hz(a){var b;a.webPlayerContextConfig?b=a.webPlayerContextConfig.rootElementId:b=a.config.attrs.id;a.elementId=b||a.elementId;"video-player"===a.elementId&&(a.elementId=a.v,a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.v:a.config.attrs.id=a.v);var c;(null==(c=a.j)?void 0:c.id)===a.elementId&&(a.elementId+="-player",a.webPlayerContextConfig?a.webPlayerContextConfig.rootElementId=a.elementId:a.config.attrs.id=a.elementId)}
k.setConfig=function(a){this.eb=a;this.config=Kz(a);Hz(this);if(!this.ia){var b;this.ia=Lz(this,(null==(b=this.config.args)?void 0:b.jsapicallback)||"onYouTubePlayerReady")}this.config.args?this.config.args.jsapicallback=null:this.config.args={jsapicallback:null};var c;if(null==(c=this.config)?0:c.attrs)a=this.config.attrs,(b=a.width)&&this.j&&(this.j.style.width=ui(Number(b)||b)),(a=a.height)&&this.j&&(this.j.style.height=ui(Number(a)||a))};
function Jz(a){if(a.config&&!0!==a.config.loaded)if(a.config.loaded=!0,!a.config.args||"0"!==a.config.args.autoplay&&0!==a.config.args.autoplay&&!1!==a.config.args.autoplay){var b;a.api.loadVideoByPlayerVars(null!=(b=a.config.args)?b:null)}else a.api.cueVideoByPlayerVars(a.config.args)}
function Mz(a){var b=!0,c=Nz(a);c&&a.config&&(a=Oz(a),b=cy(c,"version")===a);return b&&!!B("yt.player.Application.create")}
function Iz(a){if(!a.h()&&!a.V){var b=Mz(a);if(b&&"html5"===(Nz(a)?"html5":null))a.Y="html5",a.isReady()||Pz(a);else if(Qz(a),a.Y="html5",b&&a.l&&a.m)a.m.appendChild(a.l),Pz(a);else{a.config&&(a.config.loaded=!0);var c=!1;a.H=function(){c=!0;var d=Rz(a,"player_bootstrap_method")?B("yt.player.Application.createAlternate")||B("yt.player.Application.create"):B("yt.player.Application.create");var e=a.config?Kz(a.config):void 0;d&&d(a.m,e,a.webPlayerContextConfig,a.Vb);Pz(a)};
a.V=!0;b?a.H():(gy(Oz(a),a.H),(b=Sz(a))&&yz(b),Tz(a)&&!c&&z("yt.player.Application.create",null))}}}
function Nz(a){var b=Bf(a.elementId);!b&&a.j&&a.j.querySelector&&(b=a.j.querySelector("#"+a.elementId));return b}
function Pz(a){if(!a.h()){var b=Nz(a),c=!1;b&&b.getApiInterface&&b.getApiInterface()&&(c=!0);if(c){a.V=!1;if(!Rz(a,"html5_remove_not_servable_check_killswitch")){var d;if((null==b?0:b.isNotServable)&&a.config&&(null==b?0:b.isNotServable(null==(d=a.config.args)?void 0:d.video_id)))return}Uz(a)}else a.qb=kn(function(){Pz(a)},50)}}
function Uz(a){Fz(a);a.Oa=!0;var b=Nz(a);if(b){a.s=Vz(a,b,"addEventListener");a.ha=Vz(a,b,"removeEventListener");var c=b.getApiInterface();c=c.concat(b.getInternalApiInterface());for(var d=a.api,e=0;e<c.length;e++){var f=c[e];d[f]||(d[f]=Vz(a,b,f))}}for(var g in a.i)a.i.hasOwnProperty(g)&&a.s&&a.s(g,a.i[g]);Jz(a);a.ia&&a.ia(a.api);a.T.cb("onReady",a.api)}
function Vz(a,b,c){var d=b[c];return function(){var e=Ja.apply(0,arguments);try{return a.lastError=null,d.apply(b,e)}catch(f){"sendAbandonmentPing"!==c&&(f.params=c,a.lastError=f,tu(f))}}}
function Fz(a){a.Oa=!1;if(a.ha)for(var b in a.i)a.i.hasOwnProperty(b)&&a.ha(b,a.i[b]);for(var c in a.M)a.M.hasOwnProperty(c)&&window.clearTimeout(Number(c));a.M={};a.s=null;a.ha=null;b=a.api;for(var d in b)b.hasOwnProperty(d)&&(b[d]=null);b.addEventListener=function(e,f){a.addEventListener(e,f)};
b.removeEventListener=function(e,f){a.removeEventListener(e,f)};
b.destroy=function(){a.dispose()};
b.getLastError=function(){return a.getLastError()};
b.getPlayerType=function(){return a.getPlayerType()};
b.getCurrentVideoConfig=function(){return a.eb};
b.loadNewVideoConfig=function(e){a.loadNewVideoConfig(e)};
b.isReady=function(){return a.isReady()}}
k.isReady=function(){return this.Oa};
function Gz(a){a.addEventListener("WATCH_LATER_VIDEO_ADDED",function(b){Ys("WATCH_LATER_VIDEO_ADDED",b)});
a.addEventListener("WATCH_LATER_VIDEO_REMOVED",function(b){Ys("WATCH_LATER_VIDEO_REMOVED",b)})}
k.addEventListener=function(a,b){var c=this,d=Lz(this,b);d&&(0<=eb(this.Wb,a)||this.i[a]||(b=Wz(this,a),this.s&&this.s(a,b)),this.T.subscribe(a,d),"onReady"===a&&this.isReady()&&kn(function(){d(c.api)},0))};
k.removeEventListener=function(a,b){this.h()||(b=Lz(this,b))&&aj(this.T,a,b)};
function Lz(a,b){var c=b;if("string"===typeof b){if(a.ma[b])return a.ma[b];c=function(){var d=Ja.apply(0,arguments),e=B(b);if(e)try{e.apply(y,d)}catch(f){su(f)}};
a.ma[b]=c}return c?c:null}
function Wz(a,b){var c="ytPlayer"+b+a.v;a.i[b]=c;y[c]=function(d){var e=kn(function(){if(!a.h()){try{a.T.cb(b,null!=d?d:void 0)}catch(h){tu(new lo("PlayerProxy error when creating global callback",{error:h,event:b,playerId:a.v,data:d}))}var f=a.M,g=String(e);g in f&&delete f[g]}},0);
qb(a.M,String(e))};
return c}
k.getPlayerType=function(){return this.Y||(Nz(this)?"html5":null)};
k.getLastError=function(){return this.lastError};
function Qz(a){a.cancel();Fz(a);a.Y=null;a.config&&(a.config.loaded=!1);var b=Nz(a);b&&(Mz(a)||!Tz(a)?a.l=b:(b&&b.destroy&&b.destroy(),a.l=null));if(a.m)for(a=a.m;b=a.firstChild;)a.removeChild(b)}
k.cancel=function(){this.H&&my(Oz(this),this.H);window.clearTimeout(this.qb);this.V=!1};
k.L=function(){Qz(this);if(this.l&&this.config&&this.l.destroy)try{this.l.destroy()}catch(b){su(b)}this.ma=null;for(var a in this.i)this.i.hasOwnProperty(a)&&(y[this.i[a]]=null);this.eb=this.config=this.api=null;delete this.m;delete this.j;K.prototype.L.call(this)};
function Tz(a){var b,c;a=null==(b=a.config)?void 0:null==(c=b.args)?void 0:c.fflags;return!!a&&-1!==a.indexOf("player_destroy_old_version=true")}
function Oz(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.jsUrl:(a=a.config.assets)?a.js:""}
function Sz(a){return a.webPlayerContextConfig?a.webPlayerContextConfig.cssUrl:(a=a.config.assets)?a.css:""}
function Rz(a,b){if(a.webPlayerContextConfig)var c=a.webPlayerContextConfig.serializedExperimentFlags;else{var d;if(null==(d=a.config)?0:d.args)c=a.config.args.fflags}return"true"===Tm(c||"","&")[b]}
function Kz(a){for(var b={},c=r(Object.keys(a)),d=c.next();!d.done;d=c.next()){d=d.value;var e=a[d];b[d]="object"===typeof e?tb(e):e}return b}
;var Xz={},Yz="player_uid_"+(1E9*Math.random()>>>0);function Zz(a,b){var c="player",d=!1;d=void 0===d?!0:d;c="string"===typeof c?Bf(c):c;var e=Yz+"_"+Ra(c),f=Xz[e];if(f&&d)return $z(a,b)?f.api.loadVideoByPlayerVars(a.args||null):f.loadNewVideoConfig(a),f.api;f=new Ez(c,e,a,b,void 0);Xz[e]=f;Ys("player-added",f.api);Le(f,function(){delete Xz[f.getId()]});
return f.api}
function $z(a,b){return b&&b.serializedExperimentFlags?b.serializedExperimentFlags.includes("web_player_remove_playerproxy=true"):a&&a.args&&a.args.fflags?a.args.fflags.includes("web_player_remove_playerproxy=true"):!1}
;var aA=null,bA=null,cA=null;function dA(){eA()}
function fA(){eA()}
function eA(){var a=aA.getVideoData(1);a=a.title?a.title+" - YouTube":"YouTube";document.title!==a&&(document.title=a)}
function gA(){aA&&aA.sendAbandonmentPing&&aA.sendAbandonmentPing();O("PL_ATT")&&tz.dispose();for(var a=Di,b=0,c=ny.length;b<c;b++)a.Ba(ny[b]);ny.length=0;ly("//static.doubleclick.net/instream/ad_status.js");oy=!1;Km("DCLKSTAT",0);Je(cA,bA);aA&&(aA.removeEventListener("onVideoDataChange",dA),aA.destroy())}
;function hA(a,b,c){a="ST-"+sc(a).toString(36);b=b?Bc(b):"";c=c||5;dv()&&Tn(a,b,c)}
;function iA(a,b,c){b=void 0===b?{}:b;c=void 0===c?!1:c;var d=O("EVENT_ID");d&&(b.ei||(b.ei=d));if(b){d=a;var e=void 0===e?!0:e;var f=O("VALID_SESSION_TEMPDATA_DOMAINS",[]),g=wc(window.location.href);g&&f.push(g);g=wc(d);if(0<=eb(f,g)||!g&&0==d.lastIndexOf("/",0))if(P("autoescape_tempdata_url")&&(f=document.createElement("a"),mc(f,d),d=f.href),d&&(d=xc(d),f=d.indexOf("#"),d=0>f?d:d.slice(0,f)))if(e&&!b.csn&&(b.itct||b.ved)&&(b=Object.assign({csn:Ou()},b)),h){var h=parseInt(h,10);isFinite(h)&&0<h&&
hA(d,b,h)}else hA(d,b)}if(c)return!1;if((window.ytspf||{}).enabled)spf.navigate(a);else{var l=void 0===l?{}:l;var m=void 0===m?"":m;var n=void 0===n?window:n;c=n.location;a=Dc(a,l)+m;var q=void 0===q?ni:q;a:{q=void 0===q?ni:q;for(l=0;l<q.length;++l)if(m=q[l],m instanceof li&&m.Ce(a)){q=new Kb(a,Lb);break a}q=void 0}q=q||Pb;if(q instanceof Kb)var u=Mb(q);else{b:if(di){try{u=new URL(q)}catch(t){u="https:";break b}u=u.protocol}else c:{u=document.createElement("a");try{u.href=q}catch(t){u=void 0;break c}u=
u.protocol;u=":"===u||""===u?"https:":u}u="javascript:"!==u?q:void 0}void 0!==u&&(c.href=u)}return!0}
;z("yt.setConfig",Km);z("yt.config.set",Km);z("yt.setMsg",Uu);z("yt.msgs.set",Uu);z("yt.logging.errors.log",su);
z("writeEmbed",function(){var a=O("PLAYER_CONFIG");if(!a){var b=O("PLAYER_VARS");b&&(a={args:b})}nv(!0);"gvn"===a.args.ps&&(document.body.style.backgroundColor="transparent");a.attrs||(a.attrs={width:"100%",height:"100%",id:"video-player"});var c=document.referrer;b=O("POST_MESSAGE_ORIGIN");window!==window.top&&c&&c!==document.URL&&(a.args.loaderUrl=c);Nw("embed",["ol"]);c=O("WEB_PLAYER_CONTEXT_CONFIGS").WEB_PLAYER_CONTEXT_CONFIG_ID_EMBEDDED_PLAYER;if(!c.serializedForcedExperimentIds){var d=Ym(window.location.href);
d.forced_experiments&&(c.serializedForcedExperimentIds=d.forced_experiments)}var e;(null==(e=a.args)?0:e.autoplay)&&Nw("watch",["pbs","pbu","pbp"]);aA=Zz(a,c);aA.addEventListener("onVideoDataChange",dA);aA.addEventListener("onReady",fA);a=O("POST_MESSAGE_ID","player");O("ENABLE_JS_API")?cA=new gz(aA):O("ENABLE_POST_API")&&"string"===typeof a&&"string"===typeof b&&(bA=new nz(window.parent,a,b),cA=new kz(aA,bA.connection));py();P("ytidb_create_logger_embed_killswitch")||Ao();a={};Sy.h||(Sy.h=new Sy);
Sy.h.install((a.flush_logs={callback:function(){It()}},a));
zs();P("ytidb_clear_embedded_player")&&Di.ea(function(){var f,g;if(!rx){var h=ht(),l={Nc:qx,Qd:px};h.h.set(l.Nc,l);l={hd:{feedbackEndpoint:zv(kx),modifyChannelNotificationPreferenceEndpoint:zv(lx),playlistEditEndpoint:zv(mx),subscribeEndpoint:zv(ix),unsubscribeEndpoint:zv(jx),webPlayerShareEntityServiceEndpoint:zv(nx)}};var m=xv(),n={};m&&(n.client_location=m);void 0===f&&(f=On());void 0===g&&(g=h.resolve(qx));Ww(l,g,f,n);f={Nc:bx,Rd:Vw.h};h.h.set(f.Nc,f);rx=h.resolve(bx)}$x()})});
var jA=Om(function(){Rw();ov();P("embeds_web_enable_ve_logging_unification")||Hx()}),kA=Om(function(a){a.persisted||(Rw(),ov())}),lA=Om(function(a){P("embeds_web_enable_dispose_player_if_page_not_cached_killswitch")?gA():a.persisted||gA()}),mA=Om(gA);
window.addEventListener?(window.addEventListener("load",jA),window.addEventListener("pageshow",kA),window.addEventListener("pagehide",lA)):window.attachEvent&&(window.attachEvent("onload",jA),window.attachEvent("onunload",mA));z("yt.abuse.player.botguardInitialized",B("yt.abuse.player.botguardInitialized")||uz);z("yt.abuse.player.invokeBotguard",B("yt.abuse.player.invokeBotguard")||vz);z("yt.abuse.dclkstatus.checkDclkStatus",B("yt.abuse.dclkstatus.checkDclkStatus")||qy);
z("yt.player.exports.navigate",B("yt.player.exports.navigate")||iA);z("yt.util.activity.init",B("yt.util.activity.init")||Ns);z("yt.util.activity.getTimeSinceActive",B("yt.util.activity.getTimeSinceActive")||Qs);z("yt.util.activity.setTimestamp",B("yt.util.activity.setTimestamp")||Os);}).call(this);
